import matplotlib.pyplot as plt
import torch
from config import CFG
from net import net
from utils import fetch_scheduler

optimizer = torch.optim.Adam(net.parameters(), CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)

lrs=[]
x=0
for i in range(400):
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

plt.plot([i for i in range(400)],lrs)
plt.savefig('lr.png',dpi=600)