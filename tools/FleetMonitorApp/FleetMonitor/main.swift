import Cocoa
import UserNotifications
import SwiftUI
import Foundation

// ── Constants ──────────────────────────────────────────────────────
let WEIGHTS = NSString(string: "~/Projects/pact/experiments/postfilter_weights").expandingTildeInPath
let DEADLINE: Date = {
    // May 3, 2026 11:59 PM AOE (Anywhere on Earth = UTC-12)
    // = May 4, 2026 11:59 AM UTC = May 4, 2026 6:59 AM CDT
    var c = DateComponents(); c.year=2026; c.month=5; c.day=4; c.hour=11; c.minute=59
    c.timeZone = TimeZone(identifier: "UTC")
    return Calendar.current.date(from: c)!
}()
let LAST_BREAK = {
    var c = DateComponents(); c.year=2026; c.month=4; c.day=9; c.hour=10
    c.timeZone = TimeZone(identifier: "America/Chicago")
    return Calendar.current.date(from: c)!
}()
let SCORE: Double = 1.727
let LB = [(n: "PACT (ours)", s: 1.727), (n: "neural_inflate", s: 1.89), (n: "roi_v2", s: 1.94), (n: "av1_roi_lanczos", s: 1.95)]

// ── Data Model ─────────────────────────────────────────────────────
struct Experiment: Identifiable {
    let id = UUID()
    let tag: String; let epoch: Int; let scorer: Double; let age: Int; let active: Bool
}

class FleetState: ObservableObject {
    @Published var trainers = 0
    @Published var experiments: [Experiment] = []
    @Published var bestScorer: Double = 999
    var notifiedKeys: Set<String> = []

    func refresh() {
        let t = sh("ps aux | grep python | grep -E 'train_postfilter|segnet_boundary' | grep -v grep | grep -v dashboard | wc -l")
        let newCount = Int(t.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0

        // Notify on trainer death/restart
        if newCount == 0 && trainers > 0 { notify("⚠️ Training stopped", "All trainers exited") }
        if newCount > 0 && trainers == 0 && trainers != -1 { notify("✅ Training resumed", "\(newCount) trainer(s)") }
        trainers = newCount

        // Read checkpoints
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: WEIGHTS) else { return }
        var exps: [Experiment] = []
        for f in files where f.hasSuffix("_best_meta.json") {
            let path = "\(WEIGHTS)/\(f)"
            guard let data = fm.contents(atPath: path),
                  let j = try? JSONSerialization.jsonObject(with: data) as? [String:Any],
                  let scorer = j["scorer"] as? Double,
                  let epoch = j["epoch"] as? Int else { continue }
            let tag = f.replacingOccurrences(of: "postfilter_", with: "").replacingOccurrences(of: "_best_meta.json", with: "")
            if tag.contains("test") || tag.contains("debug") { continue }
            let attrs = try? fm.attributesOfItem(atPath: path)
            let mod = (attrs?[.modificationDate] as? Date) ?? .distantPast
            let age = Int(Date().timeIntervalSince(mod))
            exps.append(Experiment(tag: tag, epoch: epoch, scorer: scorer, age: age, active: age < 600))

            // Notify new best
            let key = "\(tag)_\(epoch)"
            if age < 30 && scorer < bestScorer && !notifiedKeys.contains(key) {
                bestScorer = scorer
                notifiedKeys.insert(key)
                notify("📉 New best: \(String(format: "%.4f", scorer))", "\(tag) ep \(epoch)")
            }
        }
        experiments = exps.sorted { $0.scorer < $1.scorer }
    }

    func sh(_ c: String) -> String {
        let t = Process(); let p = Pipe()
        t.standardOutput = p; t.standardError = FileHandle.nullDevice
        t.arguments = ["-c", c]; t.launchPath = "/bin/zsh"; t.launch()
        return String(data: p.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    func notify(_ title: String, _ body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.subtitle = "comma-lab"
        content.sound = UNNotificationSound.default
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }
}

// ── SwiftUI Popover ────────────────────────────────────────────────
struct FleetView: View {
    @ObservedObject var state: FleetState
    let now = Date()

    var daysLeft: Double { DEADLINE.timeIntervalSince(now) / 86400 }
    var hrsSince: Double { now.timeIntervalSince(LAST_BREAK) / 3600 }
    var active: [Experiment] { state.experiments.filter { $0.active } }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("comma-lab").font(.system(size: 13, weight: .bold, design: .monospaced)).foregroundColor(.teal)
                Spacer()
                Text(state.trainers > 0 ? "⚡ \(state.trainers) training" : "⏸ idle")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(state.trainers > 0 ? .green : .red)
            }.padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 6)

            Divider().padding(.horizontal, 8)

            // Score bar
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("SCORE").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary)
                    Text(String(format: "%.3f", SCORE)).font(.system(size: 24, weight: .bold, design: .monospaced)).foregroundColor(.teal)
                    Text("#1 · +\(String(format: "%.3f", LB[1].s - SCORE)) lead").font(.system(size: 10, design: .monospaced)).foregroundColor(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("DEADLINE").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary)
                    Text(String(format: "%.1fd", daysLeft)).font(.system(size: 24, weight: .bold, design: .monospaced)).foregroundColor(daysLeft < 5 ? .orange : .primary)
                    Text("May 3 11:59pm AOE").font(.system(size: 10, design: .monospaced)).foregroundColor(.secondary)
                }
            }.padding(.horizontal, 12).padding(.vertical, 8)

            // Progress bars
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("HEADROOM TO 1.50").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary)
                    GeometryReader { g in
                        let pct = min(1, (SCORE - 1.50) / (2.08 - 1.50))
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3).fill(Color.gray.opacity(0.2)).frame(height: 6)
                            RoundedRectangle(cornerRadius: 3).fill(Color.teal).frame(width: g.size.width * (1 - pct), height: 6)
                        }
                    }.frame(height: 6)
                    Text("\(String(format: "%.3f", SCORE - 1.50)) remaining").font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("TIME ELAPSED").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary)
                    GeometryReader { g in
                        let total: Double = 30 // ~30 day contest
                        let elapsed = total - daysLeft
                        let pct = min(1, elapsed / total)
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3).fill(Color.gray.opacity(0.2)).frame(height: 6)
                            RoundedRectangle(cornerRadius: 3).fill(daysLeft < 5 ? Color.orange : Color.blue).frame(width: g.size.width * pct, height: 6)
                        }
                    }.frame(height: 6)
                    Text("\(String(format: "%.1fh", hrsSince)) since breakthrough").font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary)
                }
            }.padding(.horizontal, 12).padding(.bottom, 8)

            Divider().padding(.horizontal, 8)

            // Active experiments
            Text("LIVE EXPERIMENTS").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary).padding(.horizontal, 12).padding(.top, 6)
            if active.isEmpty {
                Text("No active experiments").font(.system(size: 11)).foregroundColor(.red).padding(.horizontal, 12).padding(.vertical, 4)
            } else {
                ForEach(active) { e in
                    HStack {
                        Circle().fill(e.age < 120 ? Color.green : Color.orange).frame(width: 6, height: 6)
                        Text(String(format: "%.4f", e.scorer)).font(.system(size: 12, weight: .bold, design: .monospaced)).foregroundColor(.teal)
                        Text("ep \(e.epoch)").font(.system(size: 11, design: .monospaced)).foregroundColor(.secondary)
                        Spacer()
                        Text(e.tag.prefix(20)).font(.system(size: 10, design: .monospaced)).foregroundColor(.orange).lineLimit(1)
                        Text(e.age < 60 ? "\(e.age)s" : "\(e.age/60)m").font(.system(size: 10, design: .monospaced)).foregroundColor(e.age < 120 ? .green : .secondary)
                    }.padding(.horizontal, 12).padding(.vertical, 2)
                }
            }

            Divider().padding(.horizontal, 8).padding(.top, 4)

            // Leaderboard
            Text("LEADERBOARD").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary).padding(.horizontal, 12).padding(.top, 4)
            ForEach(0..<LB.count, id: \.self) { i in
                HStack {
                    Text("#\(i+1)").font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundColor(.secondary).frame(width: 20)
                    Text(String(format: "%.3f", LB[i].s)).font(.system(size: 11, weight: .bold, design: .monospaced)).foregroundColor(i == 0 ? .teal : .primary)
                    Text(LB[i].n).font(.system(size: 10, design: .monospaced)).foregroundColor(.secondary)
                    if i == 0 { Text("←").font(.system(size: 10)).foregroundColor(.teal) }
                }.padding(.horizontal, 12).padding(.vertical, 1)
            }

            Divider().padding(.horizontal, 8).padding(.top, 4)

            // Quit
            Button("Quit Fleet Monitor") { NSApplication.shared.terminate(nil) }
                .buttonStyle(.plain).font(.system(size: 11)).foregroundColor(.secondary)
                .padding(.horizontal, 12).padding(.vertical, 6)
        }
        .frame(width: 340)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

// ── App Delegate with Popover ──────────────────────────────────────
class AppDel: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    var statusItem: NSStatusItem!
    var popover: NSPopover!
    var state = FleetState()
    var timer: Timer?

    func applicationDidFinishLaunching(_ n: Notification) {
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
            if granted { print("Notifications enabled (alert-style)") }
        }
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        statusItem.button?.action = #selector(toggle)
        statusItem.button?.target = self

        popover = NSPopover()
        popover.contentSize = NSSize(width: 340, height: 460)
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(rootView: FleetView(state: state))

        state.refresh()  // cheap: just file stat
        updateTitle()
        // Light refresh every 30s (file stat only — zero CPU)
        timer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.state.refresh()
            self?.updateTitle()
        }
    }

    func updateTitle() {
        let days = DEADLINE.timeIntervalSince(Date()) / 86400
        let icon = state.trainers > 0 ? "⚡" : "⏸"
        DispatchQueue.main.async {
            self.statusItem.button?.title = "\(icon) \(String(format: "%.3f", SCORE)) · \(self.state.trainers)t · \(Int(days))d"
        }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound, .badge])
    }

    @objc func toggle() {
        if popover.isShown {
            popover.performClose(nil)
        } else if let btn = statusItem.button {
            state.refresh()
            popover.show(relativeTo: btn.bounds, of: btn, preferredEdge: .minY)
        }
    }

}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let del = AppDel()
app.delegate = del
app.run()
