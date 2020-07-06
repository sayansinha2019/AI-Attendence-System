'''
import xlwt
wb=xlwt.Workbook()
sh1=wb.add_sheet('info')
sh1.write(0,0,'SN')
sh1.write(0,1,'Name')
sh1.write(0,2,'Email')
sh1.write(0,3,'Mobile No')
sh1.write(0,4,'ID')
for i in range(1,32):
    sh1.write(0,4+i,i)'''
#wb.save(r'C:\Users\sayan\Desktop\AI Project\db.xls')
#print('Database is created Successfully')


import xlrd
rb=xlrd.open_workbook(r'C:\Users\sayan\Desktop\AI Project\db.xls')
print(rb.nsheets)
sh=rb.sheet_by_name('info')
rn=sh.nrows
print(rn)
for i in range(0,rn):
    r=sh.row_values(i)
    print(r)
import pyqrcode as pqr
from xlutils.copy import copy
wbb=copy(rb)
shd0=wbb.get_sheet(0)
op=1
while(op==1):
    name=input('please enter your name')
    email=input('enter your email id')
    mob=input('please enter your mobile number')
    idd='TN'+name[0:2].upper()+mob[0:2]+'20'
    shd0.write(rn,0,rn)
    shd0.write(rn,1,name)
    shd0.write(rn,2,email)
    shd0.write(rn,3,mob)
    shd0.write(rn,4,idd)
    url=pqr.create(idd)
    url.png(r'C:\Users\sayan\Desktop\AI Project\QRDB'+'\\'+idd+'.png',scale=10)
    op=int(input('would you like to continue'))
    rn=rn+1
wbb.save(r'C:\Users\sayan\Desktop\AI Project\db.xls')


