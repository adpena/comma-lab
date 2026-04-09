#!/usr/bin/env swift
// FleetMonitor — macOS menubar competition tracker
//
// Build & run:
//   swiftc -framework Cocoa tools/FleetMonitor/FleetMonitor.swift \
//     -o tools/FleetMonitor/FleetMonitor
//   ./tools/FleetMonitor/FleetMonitor

import Cocoa
import Foundation

// ── Competition constants ──────────────────────────────────────────
let CONTEST_DEADLINE = {
    var c = DateComponents()
    c.year = 2026; c.month = 5; c.day = 3; c.hour = 23; c.minute = 59
    c.timeZone = TimeZone(identifier: "America/Los_Angeles")
    return Calendar.current.date(from: c)!
}()

let PROMOTED_SCORE: Double = 1.727
let PROMOTED_DATE = {
    var c = DateComponents()
    c.year = 2026; c.month = 4; c.day = 9; c.hour = 10; c.minute = 0
    c.timeZone = TimeZone(identifier: "America/Chicago")
    return Calendar.current.date(from: c)!
}()

let LEADERBOARD: [(name: String, score: Double)] = [
    ("PACT (ours)", 1.727),
    ("neural_inflate", 1.89),
    ("roi_v2", 1.94),
    ("av1_roi_lanczos_unsharp", 1.95),
]

let THEORETICAL_FLOOR: Double = 1.20  // council estimate
let PRACTICAL_FLOOR: Double = 1.50    // council central estimate
let NEXT_TARGET: Double = 1.60        // council conservative target

let WEIGHTS_DIR = NSString(string: "~/Projects/pact/experiments/postfilter_weights").expandingTildeInPath

// ── App ────────────────────────────────────────────────────────────

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var timer: Timer?
    var trainers: Int = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    func refresh() {
        let now = Date()

        // Time to deadline
        let secsLeft = CONTEST_DEADLINE.timeIntervalSince(now)
        let daysLeft = secsLeft / 86400

        // Time since last breakthrough
        let sinceLast = now.timeIntervalSince(PROMOTED_DATE)
        let hoursSinceLast = sinceLast / 3600

        // Active trainers
        let ps = shell("ps aux | grep python | grep -E 'train|segnet' | grep -v grep | grep -v dashboard | wc -l").trimmingCharacters(in: .whitespacesAndNewlines)
        trainers = Int(ps) ?? 0

        // Menubar title: score + days left
        let icon = trainers > 0 ? "⚡" : "⏸"
        DispatchQueue.main.async {
            self.statusItem.button?.title = "\(icon) \(String(format: "%.3f", PROMOTED_SCORE)) · \(Int(daysLeft))d left"
        }

        // Build menu
        let menu = NSMenu()

        // ── Header ──
        let header = NSMenuItem(title: "comma-lab", action: nil, keyEquivalent: "")
        header.attributedTitle = NSAttributedString(string: "comma-lab fleet monitor", attributes: [.font: NSFont.boldSystemFont(ofSize: 13), .foregroundColor: NSColor.systemTeal])
        menu.addItem(header)
        menu.addItem(NSMenuItem.separator())

        // ── Score section ──
        addSection(menu, "SCORE")
        addRow(menu, "Official score", String(format: "%.3f", PROMOTED_SCORE))
        addRow(menu, "Lead over #2", String(format: "+%.3f", LEADERBOARD[1].score - PROMOTED_SCORE))
        addRow(menu, "Leaderboard rank", "#1 of \(LEADERBOARD.count)")
        menu.addItem(NSMenuItem.separator())

        // ── Targets ──
        addSection(menu, "HEADROOM")
        addRow(menu, "Next target", String(format: "%.2f (need %.3f)", NEXT_TARGET, PROMOTED_SCORE - NEXT_TARGET))
        addRow(menu, "Practical floor", String(format: "%.2f (council central)", PRACTICAL_FLOOR))
        addRow(menu, "Theoretical floor", String(format: "%.2f (hard limit)", THEORETICAL_FLOOR))
        addRow(menu, "Remaining headroom", String(format: "%.3f to practical", PROMOTED_SCORE - PRACTICAL_FLOOR))
        menu.addItem(NSMenuItem.separator())

        // ── Timeline ──
        addSection(menu, "TIMELINE")
        addRow(menu, "Last breakthrough", formatDuration(hoursSinceLast) + " ago")
        addRow(menu, "Breakthrough date", formatDate(PROMOTED_DATE))
        addRow(menu, "Contest deadline", formatDate(CONTEST_DEADLINE))
        addRow(menu, "Time remaining", String(format: "%.1f days", daysLeft))

        let fiveDayMark = CONTEST_DEADLINE.addingTimeInterval(-5 * 86400)
        let oneDayMark = CONTEST_DEADLINE.addingTimeInterval(-1 * 86400)
        if now < fiveDayMark {
            let toFive = fiveDayMark.timeIntervalSince(now) / 86400
            addRow(menu, "Until 5-day warning", String(format: "%.1f days", toFive))
        } else if now < oneDayMark {
            addRow(menu, "⚠️ UNDER 5 DAYS", String(format: "%.1f days left", daysLeft))
        } else {
            addRow(menu, "🔴 FINAL DAY", String(format: "%.1f hours left", secsLeft / 3600))
        }
        menu.addItem(NSMenuItem.separator())

        // ── Fleet ──
        addSection(menu, "FLEET")
        addRow(menu, "Active trainers", "\(trainers)")
        let cpuUsage = shell("ps aux | grep python | grep -E 'train|segnet' | grep -v grep | grep -v dashboard | awk '{sum+=$3} END {printf \"%.0f\", sum}'").trimmingCharacters(in: .whitespacesAndNewlines)
        addRow(menu, "Total CPU", "\(cpuUsage)%")
        let memUsage = shell("ps aux | grep python | grep -E 'train|segnet' | grep -v grep | grep -v dashboard | awk '{sum+=$6/1048576} END {printf \"%.1f\", sum}'").trimmingCharacters(in: .whitespacesAndNewlines)
        addRow(menu, "Total RAM", "\(memUsage) GB")
        menu.addItem(NSMenuItem.separator())

        // ── Leaderboard ──
        addSection(menu, "LEADERBOARD")
        for (i, entry) in LEADERBOARD.enumerated() {
            let marker = entry.name.contains("ours") ? " ← us" : ""
            addRow(menu, "#\(i+1)", String(format: "%.3f  %@%@", entry.score, entry.name, marker))
        }
        menu.addItem(NSMenuItem.separator())

        // ── Actions ──
        let dashItem = NSMenuItem(title: "Open Live Dashboard", action: #selector(openDashboard), keyEquivalent: "d")
        dashItem.target = self
        menu.addItem(dashItem)

        let siteItem = NSMenuItem(title: "Open Writeup Site", action: #selector(openSite), keyEquivalent: "s")
        siteItem.target = self
        menu.addItem(siteItem)

        let cfItem = NSMenuItem(title: "Open Cloudflare Site", action: #selector(openCF), keyEquivalent: "c")
        cfItem.target = self
        menu.addItem(cfItem)

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))

        statusItem.menu = menu
    }

    func addSection(_ menu: NSMenu, _ title: String) {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.attributedTitle = NSAttributedString(string: title, attributes: [
            .font: NSFont.systemFont(ofSize: 10, weight: .bold),
            .foregroundColor: NSColor.secondaryLabelColor,
        ])
        item.isEnabled = false
        menu.addItem(item)
    }

    func addRow(_ menu: NSMenu, _ label: String, _ value: String) {
        let item = NSMenuItem(title: "\(label):  \(value)", action: nil, keyEquivalent: "")
        item.attributedTitle = {
            let s = NSMutableAttributedString()
            s.append(NSAttributedString(string: label + "  ", attributes: [
                .font: NSFont.systemFont(ofSize: 12),
                .foregroundColor: NSColor.secondaryLabelColor,
            ]))
            s.append(NSAttributedString(string: value, attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .medium),
                .foregroundColor: NSColor.labelColor,
            ]))
            return s
        }()
        item.isEnabled = false
        menu.addItem(item)
    }

    func formatDuration(_ hours: Double) -> String {
        if hours < 1 { return "\(Int(hours * 60))m" }
        if hours < 24 { return String(format: "%.1fh", hours) }
        return String(format: "%.1f days", hours / 24)
    }

    func formatDate(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "MMM d, h:mm a"
        return f.string(from: date)
    }

    @objc func openDashboard() { NSWorkspace.shared.open(URL(string: "http://localhost:8780")!) }
    @objc func openSite() { NSWorkspace.shared.open(URL(string: "http://localhost:8767")!) }
    @objc func openCF() { NSWorkspace.shared.open(URL(string: "https://comma-lab.pages.dev")!) }

    func shell(_ command: String) -> String {
        let task = Process()
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
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
