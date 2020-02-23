# Importing program libraries (0.2)
import time
import numpy as np
from ftplib import FTP
import os

# Importing socket as sock
#-------------------------------------------------------
# Creating a function to connec to RobotStudio
# This function should be called first thing, in order to connect to RobotStudio

class RS_serial():
    def __init__(self):
        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.path = 'D:\\INSTALACION\\baselines-master\\POSITION1.txt'
        self.ftp=FTP('192.168.56.94')
        self.ftp.connect()
        self.ftp.login()


        self.cont=0
    
    def RS_connect(self):
        self.ftp=FTP('192.168.56.94')
        self.ftp.connect()
        self.ftp.login()
        print("Conexión realizada.")

        print("Línea de control enviada.")
        
#-------------------------------------------------------
# Defining a function to send information to the RobotStudio environment.

    def RS_send_info(self,x, y, z, control_str):
        
        if control_str == '0' or control_str == '2':
            x=np.interp(x, [0,10], [250,550])
            y=np.interp(y, [0,10], [-220,220])
            z=np.interp(z, [0,10], [60,390])
            
            
        else:
            
            x=float(x)*20
            y=float(y)*20
            z=float(z)*20
            
        
        #f=open(self.path,'r+',encoding = 'utf_8')
                     #Leemos el archivo del ftp
        cont='0'
        while cont=='0': 
            #f.seek(0)
            lines = []
            f=self.ftp.retrlines('RETR POSITION1.txt', lines.append)
            lines=list(filter(lambda a: a != '', lines))

            #print (lines)
            #_, _ , cont, _ = lines
            _, _ , cont = lines
            #print ('yeah')
            #_, _ , cont, _ = f.split('\n')
		
        w=open(self.path,'r+',encoding = 'utf_8')       
        w.seek(0)

        w.write("["+str(x)+','+str(y)+','+str(z)+"]\n")
        w.write(control_str +"\n")
        w.write('0\n')
        w.truncate()
        w.close()
        w=open(self.path,'rb')       
        w.seek(0)

        #print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        time.sleep(0.01)
        self.ftp.storbinary('STOR %s' % os.path.basename('POSITION1.txt'),w,1024)
        w.close()
        #print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        
        cont='0'
        while cont=='0':
          
          try:
              
              #f.seek(0)
              lines = []
              #print('--')
              f=self.ftp.retrlines('RETR POSITION1.txt', lines.append)
              lines=list(filter(lambda a: a != '', lines))
              #print(lines)
              position_s, info_confirmation, cont = lines

              time.sleep(0.05)
          except:
              try:
                time.sleep(0.01)
                #f.seek(0)
                lines = []
                f=self.ftp.retrlines('RETR POSITION1.txt', lines.append)
                lines=list(filter(lambda a: a != '', lines))
                position_s, info_confirmation, cont = lines
                #position_s, info_confirmation, cont, _ = f.split('\n')
                time.sleep(0.01)
                print("ARRANCA YA!")
              except:
                time.sleep(2)
                #f.seek(0)
                lines = []
                f=self.ftp.retrlines('RETR POSITION1.txt', lines.append)
                lines=list(filter(lambda a: a != '', lines))
                position_s, info_confirmation, cont = lines
                #position_s, info_confirmation, cont, _ = f.split('\n')
                time.sleep(0.001)
                print("ARRANCA YA POR DIOH!")
                
                
        w.close()

        return int(info_confirmation), position_s
    
    
    #-------------------------------------------------------
    # Defining a function to close the connection


    def RS_disconnect(self): 
        print("RS_disconnect")

