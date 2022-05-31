
def validate(opciones, eleccion): #ver si agregamos parametro adicional con texto a desplegar en caso de que usuario ingrese opción no valida
    # Definir validación de eleccion
    ##########################################################################

    if eleccion not in opciones:
        print('Su elección no está dentro de las opciones válidas')
        return validate(opciones, input('Ingresa una Opción: ').lower())

    # while eleccion not in opciones:
    #     print('Su elección no está dentro de las opciones válidas')
    #     eleccion = input("Ingresa una opción: ").lower()
            
    ##########################################################################
    return eleccion


if __name__ == '__main__':
    
    eleccion = input('Ingresa una Opción: ').lower()
    
    # letras = ['a','b','c','d'] # pueden probar con letras también para verificar su funcionamiento.
    numeros = ['0','1']
    # Si se ingresan valores no validos a eleccion debe seguir preguntando
    validate(numeros, eleccion)
    
