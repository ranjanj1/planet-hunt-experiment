import json, sys
n=0
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except: continue
    if d.get('type')=='assistant':
        for c in d['message']['content']:
            if c['type']=='tool_use': n+=1; print(n, 'TOOL', c['name'], (c['input'].get('description') or str(c['input'])[:100]))
            elif c['type']=='text' and c['text'].strip(): print('   TEXT', c['text'][:200].replace('\n',' '))
    if d.get('type')=='result': print('RESULT', {k:d.get(k) for k in ('subtype','duration_ms','num_turns','total_cost_usd','is_error')})
