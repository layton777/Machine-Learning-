"""
Ejercicio 4 - Elitismo Ampliado


Objetivo: modificar el manejo del elitismo para que, en lugar de
preservar solo al mejor individuo, se preserve intactos a los 3
mejores individuos de la generacion actual hacia la siguiente.

En `src/genetic_algorithm.py`, la funcion `aplicar_elitismo(poblacion,
aptitudes, n_elite)` ya esta disenada para ser generica: ordena a los
individuos por aptitud y devuelve copias de los `n_elite` mejores.
El motor `ejecutar_ag` simplemente llama a esa funcion con el parametro
`n_elite` que se le indique.

Por lo tanto, "ampliar" el elitismo NO requiere tocar el algoritmo:
basta con invocar `ejecutar_ag(..., n_elite=3)` en lugar de
`n_elite=1` (valor por defecto). Este script lo demuestra ejecutando
el problema del Ejercicio 1 con ambas configuraciones y comparando su
convergencia, para evidenciar el efecto del elitismo ampliado.


 Con n_elite=1 solo se garantiza no perder al mejor individuo; el
  resto de "buenas" soluciones puede desaparecer de una generacion a
  otra por efecto de la seleccion/cruce/mutacion.
- Con n_elite=3 se conserva mas diversidad "de calidad" entre
  generaciones, lo que normalmente produce una convergencia mas
  estable y, en problemas con multiples optimos locales, reduce el
  riesgo de perder soluciones prometedoras. Como contrapartida, un
  elitismo demasiado agresivo puede reducir la exploracion y favorecer
  la convergencia prematura en problemas mas complejos.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from genetic_algorithm import ejecutar_ag  # noqa: E402
from ejercicio1_maximizacion_cubica import RANGO_X, funcion_aptitud  # noqa: E402


CONFIGURACIONES_ELITISMO = [1, 3]


def main() -> None:
    resultados = {}

    for n_elite in CONFIGURACIONES_ELITISMO:
        resultado = ejecutar_ag(
            funcion_aptitud=funcion_aptitud,
            rangos=[RANGO_X],
            n_bits=16,
            tam_poblacion=50,
            n_generaciones=100,
            pc=0.8,
            pm=0.01,
            n_elite=n_elite,
            semilla=42,
        )
        resultados[n_elite] = resultado
        x_optimo = resultado.mejor_valor_decodificado[0]
        print(
            f"n_elite={n_elite}: mejor x={x_optimo:.5f}, "
            f"f(x)={resultado.mejor_aptitud:.5f}"
        )

    plt.figure(figsize=(8, 5))
    for n_elite, resultado in resultados.items():
        etiqueta = "Elitismo estandar (top 1)" if n_elite == 1 else "Elitismo ampliado (top 3)"
        plt.plot(resultado.historial_mejor_aptitud, label=etiqueta)
    plt.axhline(2.0, color="red", linestyle=":", label="Optimo teorico (2.0)")
    plt.title("Efecto del elitismo ampliado en la convergencia (Ejercicio 1)")
    plt.xlabel("Generacion")
    plt.ylabel("Mejor f(x) encontrado")
    plt.legend()
    plt.grid(True, alpha=0.3)

    salida = os.path.join(
        os.path.dirname(__file__), "..", "resultados", "ejercicio4_comparacion_elitismo.png"
    )
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Grafica guardada en: {salida}")
    plt.show()


if __name__ == "__main__":
    main()
