/**
 * Paces a Metal surface with CAMetalDisplayLink: the system hands over a drawable when a frame
 * drawn into it now still reaches the next refresh, and the caller shows its newest state in it.
 *
 * CAMetalDisplayLink exists only in Objective-C and Swift, so the C++ backend reaches it through
 * the few functions marked @_expose(Cxx) below, with opaque pointers and nothing else.
 */
import Foundation
import QuartzCore

@available(macOS 14.0, *)
private final class MetalDisplayPacer: NSObject, CAMetalDisplayLinkDelegate {
    private let link: CAMetalDisplayLink
    private let lock = NSLock()
    private var ready: CAMetalDrawable?
    private var runLoop: CFRunLoop?

    init(layer: CAMetalLayer) {
        link = CAMetalDisplayLink(metalLayer: layer)
        super.init()
        link.delegate = self
        // One frame between the drawable and the screen: the least a frame can wait.
        link.preferredFrameLatency = 1
        // As fast as the display goes; a display that cannot reach it runs at its own rate.
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 120, preferred: 120)
    }

    /// Runs the link on a thread of its own, so its updates never wait for the caller's event loop.
    func start() {
        let started = DispatchSemaphore(value: 0)
        let thread = Thread { [self] in
            runLoop = CFRunLoopGetCurrent()
            link.add(to: .current, forMode: .default)
            started.signal()
            CFRunLoopRun()
        }
        thread.name = "LightRHI display link"
        thread.qualityOfService = .userInteractive
        thread.start()
        started.wait()
    }

    func stop() {
        link.invalidate()
        if let runLoop {
            CFRunLoopStop(runLoop)
        }
        lock.lock()
        ready = nil
        lock.unlock()
    }

    func metalDisplayLink(_: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        // A drawable nobody took in time goes back unshown; only the newest one is worth drawing.
        lock.lock()
        ready = update.drawable
        lock.unlock()
    }

    /// The drawable for the coming refresh, once; nil until the link hands over the next one.
    func take() -> CAMetalDrawable? {
        lock.lock()
        defer {
            ready = nil
            lock.unlock()
        }
        return ready
    }
}

/// A pacer for `layer`, a CAMetalLayer; 0 where CAMetalDisplayLink does not exist (before macOS 14).
@_expose(Cxx)
public func createMetalDisplayPacer(_ layer: UnsafeMutableRawPointer) -> UInt {
    guard #available(macOS 14.0, *) else {
        return 0
    }
    let pacer = MetalDisplayPacer(layer: Unmanaged<CAMetalLayer>.fromOpaque(layer).takeUnretainedValue())
    pacer.start()
    return UInt(bitPattern: Unmanaged.passRetained(pacer).toOpaque())
}

/// The drawable for the coming refresh, retained for the caller; 0 when none is ready yet.
@_expose(Cxx)
public func takeMetalDisplayPacerDrawable(_ pacer: UInt) -> UInt {
    guard #available(macOS 14.0, *), let opaque = UnsafeMutableRawPointer(bitPattern: pacer),
          let drawable = Unmanaged<MetalDisplayPacer>.fromOpaque(opaque).takeUnretainedValue().take()
    else {
        return 0
    }
    return UInt(bitPattern: Unmanaged.passRetained(drawable as AnyObject).toOpaque())
}

/// Stops the link and releases the pacer.
@_expose(Cxx)
public func destroyMetalDisplayPacer(_ pacer: UInt) {
    guard #available(macOS 14.0, *), let opaque = UnsafeMutableRawPointer(bitPattern: pacer) else {
        return
    }
    let unmanaged = Unmanaged<MetalDisplayPacer>.fromOpaque(opaque)
    unmanaged.takeUnretainedValue().stop()
    unmanaged.release()
}
