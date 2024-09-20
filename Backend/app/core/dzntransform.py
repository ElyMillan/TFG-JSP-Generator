import os
import pickle
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from generadorFinal import JSP

def call(problema):
    global index
    with open(f"{INPUT_DIR}/{problema}/{problema}",'rb') as f:
        data:JSP = pickle.load(f)
        for t in [0, 1, 2]:
            for s0, sf, sp, s in [(2,3,1,1),(0,5,2,3),(0,5,1,5)]:
                time       = data.ProcessingTime[:,:,s0:sf:sp]
                energy     = data.EnergyConsumption[:,:,s0:sf:sp]
                precedence = np.full((data.numJobs,data.numMchs), 0)

                replace_data = {
                    "machines": data.numMchs,
                    "jobs": data.numJobs,
                    "Speed":s,
                    "time": list(time.flatten()),
                    "energy": list(energy.flatten())
                }
                
                if t == 1:
                    replace_data["releaseDate"] = [data.ReleaseDueDate[job,:,0].min() for job in range(data.numJobs)]
                    replace_data["dueDate"]     = [data.ReleaseDueDate[job,:,0].max() for job in range(data.numJobs)]
                elif t == 2:
                    replace_data["releaseDate"] = list(data.ReleaseDueDate[:,:,0].flatten())
                    replace_data["dueDate"]     = list(data.ReleaseDueDate[:,:,1].flatten())

                for job in range(data.numJobs):
                    for i, prioridad in enumerate(range(data.numMchs)):
                        precedence[job,data.Orden[job, prioridad]] = i
                replace_data["precedence"] =  list(precedence.flatten())


                new_object = data.change_rddd_type(t).select_speeds(list(range(s0,sf,sp)))

                with open(f"./Minizinc/Types/RD/type{t}.dzn","r",encoding="utf-8") as file:
                    filedata = file.read()
                    for kk,v in replace_data.items():
                        filedata = filedata.replace("{"+kk+"}",str(v))

                    os.makedirs(f"{OUTPUT_DIR}/Problema{index}/", exist_ok=True)
                    
                    with open(f"{OUTPUT_DIR}/Problema{index}/{problema}-{t}-{s}.dzn", "w+",encoding="utf-8") as new:
                        new.write(filedata)


                with open(f"{OUTPUT_DIR}/Problema{index}/Problema{index}", "wb") as new:
                    pickle.dump(new_object, new)
                index += 1

INPUT_DIR   = "./ConjuntoTFG"
OUTPUT_DIR = "./ConjuntoTFG-dzn"
os.makedirs(OUTPUT_DIR, exist_ok=True)
index   = 0

if __name__ == '__main__':
    problemas = os.listdir(INPUT_DIR)
    
    for problema in problemas:
        call(problema)
    # with ProcessPoolExecutor(cpu_count()) as executor:
    #     futures = executor.map(call, problemas)
    #     # Esperar a que todos los procesos terminen
    #     for future in as_completed(futures):
    #         try:
    #             future.result()  # Espera a que el proceso termine y maneja cualquier excepción
    #         except Exception as e:
    #             print(f"Error en una tarea: {e}")


   