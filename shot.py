import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-gl=swiftshader','--no-sandbox'])
    pg = b.new_page(viewport={'width':1920,'height':1080}); errs=[]
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type=='error' else None)
    pg.goto('file:///home/claude/flyppg/brain.html')
    for i,t in enumerate([2500, 9000, 16000]):
        pg.wait_for_timeout(t if i==0 else t-[2500,9000,16000][i-1]); pg.screenshot(path=f'shot{i+1}.png')
    print('errors:', errs[:3]); b.close()
