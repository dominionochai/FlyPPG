import math
from playwright.sync_api import sync_playwright
def route(r):
    u=r.request.url
    if 'cdnjs' in u: r.fulfill(path='/tmp/t3/package/build/three.min.js',content_type='text/javascript')
    elif 'jsdelivr' in u: r.fulfill(path='/tmp/t3/package/'+u.split('three@0.128.0/')[1],content_type='text/javascript')
    else: r.continue_()
with sync_playwright() as p:
    b=p.chromium.launch(args=['--use-gl=swiftshader','--enable-webgl','--ignore-gpu-blocklist','--no-sandbox'])
    pg=b.new_page(viewport={'width':1920,'height':1080}); errs=[]
    pg.on('pageerror',lambda e:errs.append(str(e))); pg.on('console',lambda m:errs.append(m.text) if m.type=='error' else None)
    pg.route('**/*',route); pg.goto('file:///home/claude/flyppg/brain3d.html'); pg.wait_for_timeout(2500)
    for i,(az,el,d,fi) in enumerate([(0.5,0.25,2.15,120),(1.5,0.2,2.1,300),(-0.9,0.5,1.7,430),(3.0,0.3,2.3,200)]):
        pg.evaluate(f'window.__fi={fi};window.setCam({az},{el},{d})'); pg.wait_for_timeout(2500); pg.screenshot(path=f'b3d_{i+1}.png')
    print('errors',errs[:4]); b.close()
