
from scipy.io import loadmat

path = 'cars_annos.mat'

mat = loadmat(path)

#print(list(mat))
#print('class names:', mat['class_names'][0])
#print('\n header:', mat['__header__'][0])
#print('\n version:', mat['__version__'][0])
#print('\n globals:', mat['__globals__'])
#print('\n annotations:', mat['annotations'][0])
#print(int(mat['annotations'][0][-1][-2]))

print(len(mat['annotations'][0]))


for kk in mat['annotations'][0][:50]:
    #[print(ll) for ll in kk]
    print(f"{kk[0]}: {int(kk[5])}")
