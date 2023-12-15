
import torch, torch.distributed as dist, os
from random import random

def gather_object(object):
	output_objects = [None for _ in range(dist.get_world_size())]
	dist.all_gather_object(output_objects, object)
	return [x for y in output_objects for x in y]

local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

output=[ dict(gpu=local_rank, rnd=random()) ]

output_gathered=gather_object(output)

if local_rank==0:
	print(output_gathered)

# OUTPUT
# [{'gpu': 0, 'rnd': 0.8832460463012133}, {'gpu': 1, 'rnd': 0.2782810430016065}, {'gpu': 2, 'rnd': 0.3124771564065}]
