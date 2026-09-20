import numpy as np

with open('list_eval_partition.txt', 'r') as f:
      files = f.read().splitlines()
f.close()

def Clean(data):
   clean = [x for x in data if x]
   return clean

kk = [Clean(file_name.split(' ')) for file_name in files[2:]] 

train = [m for m in kk if m[-1]=='train']

test = [m for m in kk if m[-1]!='train']
test_qry = [n for n in kk if n[-1]=='query']
test_gallery = [p for p in kk if p[-1]=='gallery']


train_labels = [int(idr[1].split('_')[-1]) for idr in train]

test_labels = [int(idt[1].split('_')[-1]) for idt in test]
test_qry_labels = [int(idq[1].split('_')[-1]) for idq in test_qry]
test_gallery_labels = [int(idg[1].split('_')[-1]) for idg in test_gallery]

print(len(train_labels), len(test_qry_labels), len(test_gallery_labels), len(test), len(test_labels))
print(train_labels[-1], test_qry_labels[-1], test_gallery_labels[-1])


def write_labels(file_name, imgs, labels):
      with open(file_name, 'w') as ff:
          for i, nn in enumerate(imgs):
               ff.write(nn[0]+','+str(labels[i])+'\n')
      ff.close()

#write_labels('train_labels.txt', train, train_labels)
write_labels('test_labels.txt', test, test_labels)
#write_labels('test_qry_labels.txt', test_qry, test_qry_labels)
#write_labels('test_gallery_labels.txt', test_gallery, test_gallery_labels)