

with open('Ebay_test.txt', 'r') as f:
     files = f.read().splitlines()
f.close()

#[print(k) for k in files[0].split(' ')]     

print(files[0])
print(files[1].split(' ')[-1])
print(files[1].split(' ')[1])


with open('Ebay_test_upd.txt', 'w') as ff:
     for name in files[1:]:
         name = name.split(' ')
         ff.write(name[-1]+','+name[1]+'\n')
ff.close()
print('Bye....')        
