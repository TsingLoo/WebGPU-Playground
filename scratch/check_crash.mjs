import puppeteer from 'puppeteer';

(async () => {
    console.log("Launching browser...");
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
    page.on('console', msg => console.log(`[BROWSER ${msg.type().toUpperCase()}]:`, msg.text()));

    console.log("Navigating...");
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 2000));
    
    console.log("Changing select element via DOM click...");
    await page.evaluate(() => {
        const selects = document.querySelectorAll('select');
        for (let s of selects) {
            if (s.options && s.options[1] && s.options[1].value === 'path tracing') {
                s.value = 'path tracing';
                s.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });

    console.log("Waiting 4s for potential crash...");
    await new Promise(r => setTimeout(r, 4000));
    
    await page.screenshot({ path: 'scratch/crash_test.jpg' });
    await browser.close();
    console.log("Done.");
})();
