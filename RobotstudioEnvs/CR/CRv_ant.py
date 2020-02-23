# Importing program libraries (0.2)
import time
import numpy as np

# Importing socket as sock
#-------------------------------------------------------
# Creating a function to connec to RobotStudio
# This function should be called first thing, in order to connect to RobotStudio

class RS_serial():
    def __init__(self):
        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.path = 'C:\\Users\\javij\\Documents\\RobotStudio\\Solutions\\Solution3\\Virtual Controllers\\IRB_120_3kg_0.58m\\HOME\\POSITION1.txt'
        self.cont=0
    
    def RS_connect(self):
        print("Conexión realizada.")

        print("Línea de control enviada.")
        
#-------------------------------------------------------
# Defining a function to send information to the RobotStudio environment.

    def RS_send_info(self,x, y, z, control_str):
        
        if control_str == '0' or control_str == '2':
            x=np.interp(x, [0,10], [340,500])
            y=np.interp(y, [0,10], [-100,190])
            z=np.interp(z, [0,10], [0,300])
            
            
        else:
            
            x=float(x)*15
            y=float(y)*15
            z=float(z)*15
            
        
        f=open(self.path,'r+',encoding = 'utf_8')
        cont='0'
        while cont=='0': 
            f.seek(0)
            _, _ , cont, _ = f.read().split('\n')
        
        f.seek(0)
        f.write("["+str(x)+','+str(y)+','+str(z)+"]\n")
        f.write(control_str +"\n")
        f.write('0\n')
        f.truncate()
       
        
        cont='0'
        while cont=='0':
          
          try:
              
              f.seek(0)
              position_s, info_confirmation, cont, _ = f.read().split('\n')
              time.sleep(0.001)
          except:
              try:
                time.sleep(0.01)
                f.seek(0)
                position_s, info_confirmation, cont, _ = f.read().split('\n')
                time.sleep(0.001)
                print("ARRANCA YA!")
              except:
                time.sleep(0.5)
                f.seek(0)
                position_s, info_confirmation, cont, _ = f.read().split('\n')
                time.sleep(0.001)
                print("ARRANCA YA POR DIOH!")
                
                
        f.close()

        return int(info_confirmation), position_s
    
    
    #-------------------------------------------------------
    # Defining a function to close the connection


    def RS_disconnect(self): 
        print("RS_disconnect")

