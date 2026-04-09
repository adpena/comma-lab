import Cocoa
import UserNotifications
import Foundation

let WEIGHTS_DIR = NSString(string: "~/Projects/pact/experiments/postfilter_weights").expandingTildeInPath
let CONTEST_END: Date = {
    var c = DateComponents(); c.year=2026; c.month=5; c.day=3; c.hour=23; c.minute=59
    c.timeZone = TimeZone(identifier: "America/Los_Angeles")
    return Calendar.current.date(from: c)!
}()
let BREAKTHROUGH_DATE: Date = {
    var c = DateComponents(); c.year=2026; c.month=4; c.day=9; c.hour=10; c.minute=0
    c.timeZone = TimeZone(identifier: "America/Chicago")
    return Calendar.current.date(from: c)!
}()
let PROMOTED: Double = 1.727
let LB2: Double = 1.89
let TARGET: Double = 1.60
let PRACTICAL_FLOOR: Double = 1.50

class App: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    var item: NSStatusItem!
    var timer: Timer?
    var bestScorer: Double = 999
    var lastTrainerCount: Int = -1
    var notifiedTags: Set<String> = []

    func applicationDidFinishLaunching(_ n: Notification) {
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { ok, _ in
            if ok { print("Notifications enabled") }
        }
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in self?.refresh() }
    }

    func refresh() {
        let now = Date()
        let daysLeft = CONTEST_END.timeIntervalSince(now) / 86400
        let hrsSince = now.timeIntervalSince(BREAKTHROUGH_DATE) / 3600

        // Count trainers (only actual python training processes)
        let trainers = countTrainers()

        // Read checkpoints
        let ckpts = readCheckpoints()
        let active = ckpts.filter { $0.age < 600 }
        let best = ckpts.min(by: { $0.scorer < $1.scorer })

        // Menubar title
        let icon = trainers > 0 ? "⚡" : "⏸"
        let scoreStr = String(format: "%.3f", PROMOTED)
        DispatchQueue.main.async {
            self.item.button?.title = "\(icon) \(scoreStr) · \(trainers)t · \(Int(daysLeft))d"
        }

        // Notifications
        if trainers == 0 && lastTrainerCount > 0 {
            notify("Training stopped", "All trainers have exited. Restart needed.")
        }
        if trainers > 0 && lastTrainerCount == 0 {
            notify("Training resumed", "\(trainers) trainer(s) running")
        }
        lastTrainerCount = trainers

        // New best checkpoint
        for c in ckpts where c.age < 30 {
            let key = "\(c.tag)_\(c.epoch)"
            if !notifiedTags.contains(key) && c.scorer < bestScorer {
                bestScorer = c.scorer
                notify("New best: \(String(format: "%.4f", c.scorer))",
                       "\(c.tag) epoch \(c.epoch)")
                notifiedTags.insert(key)
            }
        }

        // Deadline warnings
        if daysLeft < 5.01 && daysLeft > 4.99 {
            notify("⚠️ 5 days left", "Contest deadline approaching")
        }
        if daysLeft < 1.01 && daysLeft > 0.99 {
            notify("🔴 FINAL DAY", "Less than 24 hours remaining")
        }

        // Build menu
        let menu = NSMenu()
        sec(menu, "COMPETITION")
        row(menu, "Official score", String(format: "%.3f (#1)", PROMOTED))
        row(menu, "Lead over #2", String(format: "+%.3f over neural_inflate", LB2 - PROMOTED))
        row(menu, "Headroom to target", String(format: "%.3f to %.2f", PROMOTED - TARGET, TARGET))
        row(menu, "Practical floor", String(format: "%.2f (council est.)", PRACTICAL_FLOOR))

        menu.addItem(NSMenuItem.separator())
        sec(menu, "TIMELINE")
        row(menu, "Days remaining", String(format: "%.1f", daysLeft))
        row(menu, "Since breakthrough", hrsSince < 24 ? String(format: "%.1fh", hrsSince) : String(format: "%.1fd", hrsSince/24))
        let df = DateFormatter(); df.dateFormat = "MMM d, h:mm a"
        row(menu, "Deadline", df.string(from: CONTEST_END))

        if daysLeft < 5 { row(menu, "⚠️ WARNING", String(format: "%.1f days left!", daysLeft)) }

        menu.addItem(NSMenuItem.separator())
        sec(menu, "LIVE EXPERIMENTS (\(active.count) active)")
        for c in active.sorted(by: { $0.scorer < $1.scorer }) {
            let age = c.age < 60 ? "\(c.age)s" : "\(c.age/60)m"
            let trend = c.trend != 0 ? String(format: " Δ%.4f/ep", c.trend) : ""
            let eta = c.etaEpochs.map { $0 < 9999 ? " ~\($0)ep to proxy" : "" } ?? ""
            row(menu, String(format: "%.4f", c.scorer), "ep \(c.epoch) · \(c.tag) · \(age)\(trend)\(eta)")
        }
        if active.isEmpty { row(menu, "--", "no active experiments") }

        menu.addItem(NSMenuItem.separator())
        sec(menu, "FLEET")
        row(menu, "Trainers", "\(trainers)")
        if trainers == 0 {
            let warn = NSMenuItem(title: "  ⚠️ NO TRAINERS — restart needed", action: nil, keyEquivalent: "")
            warn.isEnabled = false
            menu.addItem(warn)
        }

        menu.addItem(NSMenuItem.separator())
        sec(menu, "ALL CHECKPOINTS")
        for c in ckpts.prefix(8) {
            let age = c.age < 60 ? "\(c.age)s" : c.age < 3600 ? "\(c.age/60)m" : "\(c.age/3600)h"
            let fresh = c.age < 600 ? "●" : "○"
            row(menu, "\(fresh) \(String(format: "%.4f", c.scorer))", "ep \(c.epoch) · \(c.tag) · \(age)")
        }

        menu.addItem(NSMenuItem.separator())
        let q = NSMenuItem(title: "Quit Fleet Monitor", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(q)

        item.menu = menu
    }

    struct Checkpoint {
        let tag: String; let epoch: Int; let scorer: Double; let age: Int
        let trend: Double; let etaEpochs: Int?
    }

    func readCheckpoints() -> [Checkpoint] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: WEIGHTS_DIR) else { return [] }
        var results: [Checkpoint] = []
        for f in files where f.hasSuffix("_best_meta.json") {
            let path = "\(WEIGHTS_DIR)/\(f)"
            guard let data = fm.contents(atPath: path),
                  let j = try? JSONSerialization.jsonObject(with: data) as? [String:Any],
                  let scorer = j["scorer"] as? Double,
                  let epoch = j["epoch"] as? Int else { continue }
            let tag = f.replacingOccurrences(of: "postfilter_", with: "").replacingOccurrences(of: "_best_meta.json", with: "")
            let attrs = try? fm.attributesOfItem(atPath: path)
            let mod = (attrs?[.modificationDate] as? Date) ?? .distantPast
            let age = Int(Date().timeIntervalSince(mod))
            // Skip test/debug checkpoints
            if tag.contains("test") || tag.contains("debug") { continue }
            results.append(Checkpoint(tag: tag, epoch: epoch, scorer: scorer, age: age, trend: 0, etaEpochs: nil))
        }
        return results.sorted { $0.scorer < $1.scorer }
    }

    func countTrainers() -> Int {
        let out = sh("ps aux | grep python | grep -E 'train_postfilter|segnet_boundary' | grep -v grep | grep -v dashboard | wc -l")
        return Int(out.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }

    func notify(_ title: String, _ body: String) {
        let c = UNMutableNotificationContent()
        c.title = title; c.body = body; c.sound = .default
        let r = UNNotificationRequest(identifier: UUID().uuidString, content: c, trigger: nil)
        UNUserNotificationCenter.current().add(r)
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent n: UNNotification, withCompletionHandler h: @escaping (UNNotificationPresentationOptions) -> Void) {
        h([.banner, .sound])
    }

    func sec(_ m: NSMenu, _ t: String) {
        let i = NSMenuItem(title: t, action: nil, keyEquivalent: "")
        i.attributedTitle = NSAttributedString(string: t, attributes: [.font: NSFont.systemFont(ofSize: 10, weight: .bold), .foregroundColor: NSColor.secondaryLabelColor])
        i.isEnabled = false; m.addItem(i)
    }

    func row(_ m: NSMenu, _ l: String, _ v: String) {
        let i = NSMenuItem(title: "\(l): \(v)", action: nil, keyEquivalent: "")
        let s = NSMutableAttributedString()
        s.append(NSAttributedString(string: "\(l)  ", attributes: [.font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.secondaryLabelColor]))
        s.append(NSAttributedString(string: v, attributes: [.font: NSFont.monospacedSystemFont(ofSize: 12, weight: .medium), .foregroundColor: NSColor.labelColor]))
        i.attributedTitle = s; i.isEnabled = false; m.addItem(i)
    }

    func sh(_ c: String) -> String {
        let t = Process(); let p = Pipe()
        t.standardOutput = p; t.standardError = FileHandle.nullDevice
        t.arguments = ["-c", c]; t.launchPath = "/bin/zsh"; t.launch()
        return String(data: p.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let d = App()
app.delegate = d
app.run()
