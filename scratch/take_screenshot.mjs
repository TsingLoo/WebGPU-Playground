import puppeteer from 'puppeteer';

(async () => {
    console.log("Launching browser...");
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    console.log("Navigating to http://localhost:5173...");
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    
    await new Promise(r => setTimeout(r, 1000));
    
    // Switch to Wavefront PT
    console.log("Switching to pt_wavefront...");
    await page.evaluate(() => {
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

    console.log("Waiting 3s for render...");
    await new Promise(r => setTimeout(r, 3000));
    
    await page.screenshot({ path: 'scratch/screenshot.png' });
    console.log("Screenshot saved.");

    await browser.close();
    console.log("Done.");
})();
