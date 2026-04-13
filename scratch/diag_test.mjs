import puppeteer from 'puppeteer';

(async () => {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--enable-unsafe-webgpu']
    });
    const page = await browser.newPage();

    page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('[vite]') || text.includes('powerPreference') || text.includes('404')) return;
        console.log(`[${msg.type().toUpperCase()}]`, text);
    });

    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 3000));

    // Switch to path tracing
    await page.evaluate(() => {
        const selects = document.querySelectorAll('select');
        for (let s of selects) {
            for (let opt of s.options) {
                if (opt.value === 'path tracing') {
                    s.value = 'path tracing';
                    s.dispatchEvent(new Event('change', { bubbles: true }));
                    return; // ONLY FIRE ONCE
                }
            }
        }
    });

    // Wait enough for frames and the readback
    await new Promise(r => setTimeout(r, 20000));
    
    await browser.close();
})();
