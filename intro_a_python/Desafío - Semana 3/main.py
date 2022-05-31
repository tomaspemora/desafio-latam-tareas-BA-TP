
# No modificar
from verify import verificar
import preguntas as p
from question import choose_q
from print_preguntas import print_pregunta
from level import choose_level
from validador import validate
import time
import os
import sys

# valores iniciales - 
n_pregunta = 0
continuar = 'y'
correcto = True
p_level = 10
op_sys = 'cls' if sys.platform == 'win32' else 'clear'


# ----------------------------------------

os.system(op_sys) # limpiar pantalla

print('Bienvenido a la Trivia')
opcion = input('''Ingrese una opción para Jugar!
        1. Jugar
        0. Salir
        
    > ''')
# 1. validar opcion
opciones_validas = ["0","1"]
opcion = validate(opciones_validas,opcion)

# 2. Definir el comportamiento de Salir
if opcion == '0':
    print("Nos vemos pronto!")
    time.sleep(2)
    os.system(op_sys)
    exit()
    # finalizar programa
    
print("Jugar")
# Funcionamiento de preguntas
while correcto and n_pregunta < 3*p_level:
    
    if n_pregunta == 0:
        p_level = input('¿Cuántas preguntas por nivel? (Máximo 3): ')
        # 3. Validar el número de preguntas por nivel
        niveles_validos = ["1","2","3"]
        p_level = int(validate(niveles_validos,p_level))
    if continuar == 'y':
        #contador de preguntas
        n_pregunta += 1
        # 4. Escoger el nivel de la pregunta
        nivel = choose_level(n_pregunta,p_level)
        print(f'Pregunta {n_pregunta}:')
        # 5. Escoger el enunciado y las alternativas de una pregunta según el nivel escogido
        enunciado, alternativas = choose_q(nivel)
        #6. Imprimir el enunciado y sus alternativas en pantalla
        print_pregunta(enunciado,alternativas)
        respuesta = input('Escoja la alternativa correcta:\n> ').lower()
        # 7. Validar la respuesta entregada
        respuestas_validas = ["a", "b", "c", "d"]
        respuesta = validate(respuestas_validas,respuesta)
        # 8. Verificar si la respuesta es correcta o no
        correcto = verificar(alternativas,respuesta)
        if correcto and n_pregunta < 3*p_level:
            print('Muy bien sigue así!')
            continuar = input('Desea continuar? [y/n]: ').lower()
            #9. Validar si es que se responde y o n
            continuar_valido= ["y","n"]
            continuar = validate(continuar_valido,continuar)
            os.system(op_sys)
        elif correcto and n_pregunta == 3*p_level:
            print(f'Felicitaciones, Has respondido {3*p_level} preguntas correctas. \n Has ganado la Trivia \n Gracias por Jugar, hasta luego!!!')
            time.sleep(3)
            os.system(op_sys)
            exit()
        else: 
            print(f'Lo siento, conseguiste {n_pregunta - 1} respuestas correctas,\n Sigue participando!!')
            time.sleep(3)
            exit()
    else: 
        print('Nos vemos la proxima vez, sigue practicando')
        time.sleep(3)
        exit()
            
            
