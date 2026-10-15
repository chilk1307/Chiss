p='/content/SEARLE/src/datasets.py'
s=open(p).read()
a=''.join('        '+l.strip()+'\n' for l in open('/content/fiq_add.txt').read().split('\n') if l.strip())
k='        print(f"FashionIQ {split}'
assert s.count(k)==1
open(p,'w').write(s.replace(k,a+k))
print('patched')
