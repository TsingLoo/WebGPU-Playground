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
    
    console.log("Waiting for 1 second...");
    await new Promise(r => setTimeout(r, 1000));
    
    // Switch to Wavefront PT
    console.log("Switching to pt_wavefront...");
    await page.evaluate(() => {
        // find select with option pt_wavefront
        const selects = document.querySelectorAll('select');
        for (const select of selects) {
            for (const option of select.options) {
                if (option.value === 'pt_wavefront') {
                    select.value = 'pt_wavefront';
                    select.dispatchEvent(new Event('change'));
                    return;
                }
            }
        }
    });

    await new Promise(r => setTimeout(r, 3000)); // wait for rendering and validation errors
    await browser.close();
    console.log("Done.");
})();
