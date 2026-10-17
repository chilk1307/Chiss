import torch,pickle,sys,os
R='/content/SEARLE/data/oti_pseudo_tokens/imagenet/test/'
out=sys.argv[1]
n=[];t=[];e=[]
for k in range(3):
 d=R+'oti_B32_s%d/'%k
 n+=pickle.load(open(d+'image_names.pkl','rb'))
 t.append(torch.load(d+'oti_pseudo_tokens.pt',map_location='cpu'))
 e.append(torch.load(d+'ema_oti_pseudo_tokens.pt',map_location='cpu'))
assert len(n)==len(set(n))
if out=='oti_B32': assert len(n)==100000
os.makedirs(R+out,exist_ok=True)
pickle.dump(n,open(R+out+'/image_names.pkl','wb'))
torch.save(torch.vstack(t),R+out+'/oti_pseudo_tokens.pt')
torch.save(torch.vstack(e),R+out+'/ema_oti_pseudo_tokens.pt')
print(len(n))
