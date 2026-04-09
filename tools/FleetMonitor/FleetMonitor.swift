#!/usr/bin/env swift
// FleetMonitor — macOS menubar app for comma-lab training
//
// Shows best score in menubar, sends notifications on improvement,
// lists active trainers in dropdown.
//
// Build & run:
//   swiftc -framework Cocoa -framework UserNotifications \
//     tools/FleetMonitor/FleetMonitor.swift -o tools/FleetMonitor/FleetMonitor
//   ./tools/FleetMonitor/FleetMonitor

import Cocoa
import Foundation

let WEIGHTS_DIR = NSString(string: "~/Projects/pact/experiments/postfilter_weights").expandingTildeInPath

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var timer: Timer?
    var bestScorer: Double = 999.0
    var bestTag: String = "--"
    var bestEpoch: Int = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create menubar item
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "⏳ --"
        statusItem.button?.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .medium)

        updateMenu()

        // Poll every 5 seconds
        timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
        refresh()
    }

    func refresh() {
        // Read checkpoint metadata files
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: WEIGHTS_DIR) else { return }

        var checkpoints: [(tag: String, epoch: Int, scorer: Double, age: Int)] = []

        for file in files where file.hasSuffix("_best_meta.json") {
            let path = "\(WEIGHTS_DIR)/\(file)"
            guard let data = fm.contents(atPath: path),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let scorer = json["scorer"] as? Double,
                  let epoch = json["epoch"] as? Int else { continue }

            let tag = file.replacingOccurrences(of: "postfilter_", with: "")
                         .replacingOccurrences(of: "_best_meta.json", with: "")
            let attrs = try? fm.attributesOfItem(atPath: path)
            let mod = (attrs?[.modificationDate] as? Date) ?? Date.distantPast
            let age = Int(Date().timeIntervalSince(mod))

            checkpoints.append((tag: tag, epoch: epoch, scorer: scorer, age: age))
        }

        checkpoints.sort { $0.scorer < $1.scorer }

        if let best = checkpoints.first {
            // Update menubar title
            let display = String(format: "%.3f", best.scorer)
            let icon = best.age < 120 ? "🟢" : "🔵"
            DispatchQueue.main.async {
                self.statusItem.button?.title = "\(icon) \(display)"
            }

            // Track improvement
            if best.scorer < self.bestScorer - 0.001 {
                print("NEW BEST: \(display) — \(best.tag) ep \(best.epoch)")
                self.bestScorer = best.scorer
                self.bestTag = best.tag
                self.bestEpoch = best.epoch
            } else if self.bestScorer > 900 {
                self.bestScorer = best.scorer
                self.bestTag = best.tag
                self.bestEpoch = best.epoch
            }
        }

        // Count active training processes
        let ps = shell("ps aux | grep python | grep -E 'train|segnet' | grep -v grep | grep -v dashboard | wc -l").trimmingCharacters(in: .whitespacesAndNewlines)
        let trainerCount = Int(ps) ?? 0

        DispatchQueue.main.async {
            self.updateMenu(checkpoints: checkpoints, trainers: trainerCount)
        }
    }

    func updateMenu(checkpoints: [(tag: String, epoch: Int, scorer: Double, age: Int)] = [], trainers: Int = 0) {
        let menu = NSMenu()

        menu.addItem(NSMenuItem(title: "comma-lab fleet monitor", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())

        menu.addItem(NSMenuItem(title: "Promoted floor: 1.727", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "Active trainers: \(trainers)", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())

        menu.addItem(NSMenuItem(title: "CHECKPOINTS", action: nil, keyEquivalent: ""))
        for c in checkpoints.prefix(8) {
            let ageStr = c.age < 60 ? "\(c.age)s" : c.age < 3600 ? "\(c.age/60)m" : "\(c.age/3600)h"
            let item = NSMenuItem(
                title: "  \(String(format: "%.4f", c.scorer))  ep \(c.epoch)  \(c.tag)  (\(ageStr))",
                action: nil, keyEquivalent: ""
            )
            item.isEnabled = false
            menu.addItem(item)
        }

        menu.addItem(NSMenuItem.separator())

        let dashItem = NSMenuItem(title: "Open Dashboard", action: #selector(openDashboard), keyEquivalent: "d")
        dashItem.target = self
        menu.addItem(dashItem)

        let siteItem = NSMenuItem(title: "Open Writeup Site", action: #selector(openSite), keyEquivalent: "s")
        siteItem.target = self
        menu.addItem(siteItem)

        menu.addItem(NSMenuItem.separator())
        let quitItem = NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    @objc func openDashboard() {
        NSWorkspace.shared.open(URL(string: "http://localhost:8780")!)
    }

    @objc func openSite() {
        NSWorkspace.shared.open(URL(string: "http://localhost:8767")!)
    }

    func shell(_ command: String) -> String {
        let task = Process()
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        task.arguments = ["-c", command]
        task.launchPath = "/bin/zsh"
        task.launch()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? ""
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
