import puppeteer from 'puppeteer';

(async () => {
    console.log("Launching browser...");
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    // Capture page errors
    page.on('pageerror', error => {
        console.error('PAGE ERROR:', error.message);
    });

    // Capture console messages
    page.on('console', msg => {
        if (msg.type() === 'error' || msg.type() === 'warning') {
            console.log(`[BROWSER ${msg.type().toUpperCase()}]:`, msg.text());
        }
    });

    console.log("Navigating to http://localhost:5173...");
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    
    console.log("Waiting for 3 seconds...");
    await new Promise(r => setTimeout(r, 3000));
    
    // Switch to Wavefront PT
    await page.evaluate(() => {
        const select = document.querySelector('select');
        if (select) {
            // Usually dat.gui elements are hard to find natively, but we can override window properties
            // Or maybe the user says "pt_wavefront"
        }
    });

    // The quickest way is to inject an uncapturederror listener
    await page.evaluate(() => {
        const gpuConfig = navigator.gpu;
        if (!gpuConfig) return;
        window.addEventListener('unhandledrejection', event => {
            console.error('Unhandled Promise Rejection:', event.reason);
        });
        
        // Find canvas device
        // Since we don't have access to `device` directly, we rely on console errors.
    });

    await new Promise(r => setTimeout(r, 2000)); // wait for rendering
    await browser.close();
    console.log("Done.");
})();
