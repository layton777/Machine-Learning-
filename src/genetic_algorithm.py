"""
genetic_algorithm.py

Implementacion base y reutilizable de un Algoritmo Genetico (AG) con
codificacion binaria. Este modulo es la "pieza comun" que todos los
ejercicios de la actividad importan y configuran segun sus necesidades
(rango de busqueda, funcion de aptitud, numero de variables, tasa de
mutacion, numero de individuos preservados por elitismo, etc.).

Diseno general

Cada individuo es una cadena de bits (lista de 0s y 1s).
`decode` transforma esa cadena de bits en uno o varios numeros reales
dentro de un rango definido (esto resuelve el mapeo binario -> real
que piden los ejercicios 1 y 2).
La funcion de aptitud (fitness) es siempre inyectada desde fuera, asi
el mismo motor sirve para maximizar o minimizar cualquier funcion.
La seleccion es por torneo, el cruce es de un punto y la mutacion es
"bit flip" con probabilidad `pm`.
El elitismo es configurable: `n_elite` define cuantos de los mejores
individuos de la generacion actual pasan intactos a la siguiente
(por defecto 1, pero el Ejercicio 4 lo cambia a 3).

Todas las funciones estan documentadas con docstrings estilo Google para
que el README y cualquier herramienta de documentacion automatica puedan
aprovecharlas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple


Individuo = List[int]  # Un individuo es una lista de bits (0/1)


# Codificacion, se pasa a Decodificacion binaria

def decodificar_variable(bits: Sequence[int], rango: Tuple[float, float]) -> float:
    """Convierte una subcadena de bits en un numero real dentro de `rango`.

    Args:
        bits: subcadena binaria (ej. [1, 0, 1, 1]) correspondiente a UNA
            variable del cromosoma.
        rango: tupla (min, max) del espacio de busqueda para esa variable.

    Returns:
        El valor real decodificado, mapeado linealmente desde el rango
        binario [0, 2^n - 1] hacia [min, max].
    """
    n = len(bits)
    entero = int("".join(str(b) for b in bits), 2) if n > 0 else 0
    maximo_entero = 2 ** n - 1
    minimo, maximo = rango
    if maximo_entero == 0:
        return minimo
    return minimo + (entero / maximo_entero) * (maximo - minimo)


def decodificar_cromosoma(
    cromosoma: Individuo, rangos: Sequence[Tuple[float, float]]
) -> List[float]:
    """Decodifica un cromosoma completo que puede representar N variables.

    El cromosoma se divide en partes iguales segun la cantidad de rangos
    entregados. Por ejemplo, para el Ejercicio 2 (x, y) con un cromosoma
    de 8 bits y `rangos=[(min_x, max_x), (min_y, max_y)]`, los primeros
    4 bits se asignan a x y los ultimos 4 a y.

    Args:
        cromosoma: cromosoma binario completo.
        rangos: lista de rangos (min, max), uno por variable.

    Returns:
        Lista con los valores reales decodificados, en el mismo orden
        que `rangos`.
    """
    n_variables = len(rangos)
    largo_total = len(cromosoma)
    largo_por_variable = largo_total // n_variables

    valores = []
    for i, rango in enumerate(rangos):
        inicio = i * largo_por_variable
        # La ultima variable absorbe los bits sobrantes si la division
        # no es exacta.
        fin = largo_total if i == n_variables - 1 else inicio + largo_por_variable
        sub_bits = cromosoma[inicio:fin]
        valores.append(decodificar_variable(sub_bits, rango))
    return valores



# Operadores geneticos

def crear_individuo(n_bits: int) -> Individuo:
    """Genera un individuo aleatorio de `n_bits` bits."""
    return [random.randint(0, 1) for _ in range(n_bits)]


def crear_poblacion(tam_poblacion: int, n_bits: int) -> List[Individuo]:
    """Genera una poblacion inicial aleatoria."""
    return [crear_individuo(n_bits) for _ in range(tam_poblacion)]


def seleccion_torneo(
    poblacion: List[Individuo], aptitudes: List[float], k: int = 3
) -> Individuo:
    """Selecciona un individuo mediante torneo de tamano `k`.

    Se eligen `k` individuos al azar y se retorna una copia del que tenga
    mejor aptitud (mayor valor). Esto asume que la funcion de aptitud ya
    esta orientada a "mayor es mejor" (para minimizar, se debe invertir
    el signo dentro de la funcion de aptitud, tal como se hace en el
    Ejercicio 2).
    """
    participantes = random.sample(range(len(poblacion)), k)
    mejor_idx = max(participantes, key=lambda idx: aptitudes[idx])
    return list(poblacion[mejor_idx])


def cruce_un_punto(
    padre1: Individuo, padre2: Individuo, pc: float = 0.8
) -> Tuple[Individuo, Individuo]:
    """Cruce de un punto entre dos padres, con probabilidad `pc`.

    Si no ocurre cruce (probabilidad 1 - pc), los hijos son copias
    identicas de los padres.
    """
    if random.random() > pc or len(padre1) < 2:
        return list(padre1), list(padre2)

    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


def mutacion_bit_flip(individuo: Individuo, pm: float) -> Individuo:
    """Aplica mutacion "bit flip": cada bit puede invertirse con prob. `pm`."""
    return [bit ^ 1 if random.random() < pm else bit for bit in individuo]


def aplicar_elitismo(
    poblacion: List[Individuo], aptitudes: List[float], n_elite: int
) -> List[Individuo]:
    """Devuelve una copia intacta de los `n_elite` mejores individuos.

    Esta es la funcion clave para el Ejercicio 4: por defecto `n_elite=1`
    preserva solo al mejor individuo, pero basta con invocar esta misma
    funcion con `n_elite=3` para preservar a los 3 mejores sin modificar
    el resto del algoritmo.

    Args:
        poblacion: poblacion actual.
        aptitudes: aptitud de cada individuo (mismo orden que `poblacion`).
        n_elite: cantidad de individuos elite a preservar intactos.

    Returns:
        Lista con copias de los `n_elite` mejores individuos, ordenados
        de mejor a peor aptitud.
    """
    indices_ordenados = sorted(
        range(len(poblacion)), key=lambda idx: aptitudes[idx], reverse=True
    )
    mejores_indices = indices_ordenados[:n_elite]
    return [list(poblacion[idx]) for idx in mejores_indices]



# Motor principal del Algoritmo Genetico


@dataclass
class ResultadoAG:
    """Contenedor con los resultados de una corrida del AG."""

    mejor_individuo: Individuo
    mejor_valor_decodificado: List[float]
    mejor_aptitud: float
    historial_mejor_aptitud: List[float] = field(default_factory=list)
    historial_aptitud_promedio: List[float] = field(default_factory=list)


def ejecutar_ag(
    funcion_aptitud: Callable[[List[float]], float],
    rangos: Sequence[Tuple[float, float]],
    n_bits: int = 16,
    tam_poblacion: int = 50,
    n_generaciones: int = 100,
    pc: float = 0.8,
    pm: float = 0.01,
    n_elite: int = 1,
    semilla: int | None = None,
) -> ResultadoAG:
    """Ejecuta el algoritmo genetico completo.

    Args:
        funcion_aptitud: funcion que recibe la lista de variables ya
            decodificadas (ej. [x] o [x, y]) y retorna un valor de
            aptitud a MAXIMIZAR. Para minimizar una funcion, se debe
            retornar su valor negativo (ver Ejercicio 2).
        rangos: lista de rangos (min, max) por variable. Su longitud
            define cuantas variables tiene el cromosoma.
        n_bits: largo total del cromosoma binario.
        tam_poblacion: cantidad de individuos por generacion.
        n_generaciones: numero de generaciones a evolucionar.
        pc: probabilidad de cruce.
        pm: probabilidad de mutacion por bit.
        n_elite: cantidad de mejores individuos preservados sin cambios
            entre generaciones (elitismo).
        semilla: semilla opcional para reproducibilidad.

    Returns:
        Un objeto `ResultadoAG` con el mejor individuo encontrado, su
        valor decodificado, su aptitud y el historial de convergencia
        (mejor aptitud y aptitud promedio por generacion), util para
        graficar.
    """
    if semilla is not None:
        random.seed(semilla)

    poblacion = crear_poblacion(tam_poblacion, n_bits)
    historial_mejor: List[float] = []
    historial_promedio: List[float] = []

    mejor_individuo_global = None
    mejor_aptitud_global = float("-inf")

    for _ in range(n_generaciones):
        # 1. Evaluar aptitud de toda la poblacion
        aptitudes = [
            funcion_aptitud(decodificar_cromosoma(ind, rangos)) for ind in poblacion
        ]

        # 2. Registrar estadisticas de esta generacion (para graficar)
        mejor_gen_idx = max(range(len(poblacion)), key=lambda i: aptitudes[i])
        mejor_aptitud_gen = aptitudes[mejor_gen_idx]
        historial_mejor.append(mejor_aptitud_gen)
        historial_promedio.append(sum(aptitudes) / len(aptitudes))

        if mejor_aptitud_gen > mejor_aptitud_global:
            mejor_aptitud_global = mejor_aptitud_gen
            mejor_individuo_global = list(poblacion[mejor_gen_idx])

        # 3. Elitismo: los n_elite mejores pasan intactos
        nueva_poblacion = aplicar_elitismo(poblacion, aptitudes, n_elite)

        # 4. Completar el resto de la poblacion con seleccion + cruce + mutacion
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo1, hijo2 = cruce_un_punto(padre1, padre2, pc)
            hijo1 = mutacion_bit_flip(hijo1, pm)
            hijo2 = mutacion_bit_flip(hijo2, pm)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion[:tam_poblacion]

    mejor_valor_decodificado = decodificar_cromosoma(mejor_individuo_global, rangos)

    return ResultadoAG(
        mejor_individuo=mejor_individuo_global,
        mejor_valor_decodificado=mejor_valor_decodificado,
        mejor_aptitud=mejor_aptitud_global,
        historial_mejor_aptitud=historial_mejor,
        historial_aptitud_promedio=historial_promedio,
    )
