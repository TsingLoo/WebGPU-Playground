import puppeteer from 'puppeteer';
(async () => {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
    page.on('console', msg => console.log('[BROWSER log]', msg.text()));
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 2000));
    await page.evaluate(() => {
        const selects = document.querySelectorAll('select');
        for (let s of selects) {
            for (let opt of s.options) {
                if (opt.value === 'path tracing') {
                    s.value = 'path tracing';
                    s.dispatchEvent(new Event('change', { bubbles: true }));
                    console.log('DOM CLICK INJECTED!');
                }
            }
        }
    });
    await new Promise(r => setTimeout(r, 4000));
    await browser.close();
})();
