"""
Ejercicio 3 - Impacto de la Tasa de Mutacion

Objetivo: ejecutar el AG del Ejercicio 1 (maximizacion de
f(x) = x^3 - 4x^2 + 5x) tres veces, cambiando unicamente la
probabilidad de mutacion `pm`, y comparar la convergencia resultante.

Valores de pm evaluados: 0.01, 0.1 y 0.5

Lo que espero observar

- pm = 0.01: deberia producir una convergencia mas estable.
- pm = 0.1: agrega mas variedad sin cambiar demasiado el progreso.
- pm = 0.5: puede ser demasiado alta y hacer que la busqueda sea mas
    aleatoria.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from genetic_algorithm import ejecutar_ag  # noqa: E402
from ejercicio1_maximizacion_cubica import RANGO_X, funcion_aptitud  # noqa: E402


TASAS_MUTACION = [0.01, 0.1, 0.5]


def main() -> None:
    resultados = {}

    for pm in TASAS_MUTACION:
        resultado = ejecutar_ag(
            funcion_aptitud=funcion_aptitud,
            rangos=[RANGO_X],
            n_bits=16,
            tam_poblacion=50,
            n_generaciones=100,
            pc=0.8,
            pm=pm,
            n_elite=1,
            semilla=42,  # misma semilla para que la unica variable sea pm
        )
        resultados[pm] = resultado
        x_optimo = resultado.mejor_valor_decodificado[0]
        print(f"pm={pm:>4}: mejor x={x_optimo:.5f}, f(x)={resultado.mejor_aptitud:.5f}")

    # --- Grafica comparativa: las 3 curvas de convergencia juntas ---
    plt.figure(figsize=(9, 6))
    for pm, resultado in resultados.items():
        plt.plot(resultado.historial_mejor_aptitud, label=f"pm = {pm}")
    plt.axhline(2.0, color="red", linestyle=":", label="Optimo teorico (2.0)")
    plt.title("Impacto de la tasa de mutacion en la convergencia (Ejercicio 1)")
    plt.xlabel("Generacion")
    plt.ylabel("Mejor f(x) encontrado")
    plt.legend()
    plt.grid(True, alpha=0.3)

    carpeta_resultados = os.path.join(os.path.dirname(__file__), "..", "resultados")
    salida_comparativa = os.path.join(
        carpeta_resultados, "ejercicio3_comparacion_pm.png"
    )
    plt.savefig(salida_comparativa, dpi=150, bbox_inches="tight")
    print(f"Grafica comparativa guardada en: {salida_comparativa}")

    # --- Ademas, una grafica individual por cada pm (como pide el enunciado) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (pm, resultado) in zip(axes, resultados.items()):
        ax.plot(resultado.historial_mejor_aptitud, label="Mejor aptitud")
        ax.plot(
            resultado.historial_aptitud_promedio,
            linestyle="--",
            label="Aptitud promedio",
        )
        ax.axhline(2.0, color="red", linestyle=":")
        ax.set_title(f"pm = {pm}")
        ax.set_xlabel("Generacion")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("f(x)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Convergencia individual por tasa de mutacion")

    salida_individuales = os.path.join(
        carpeta_resultados, "ejercicio3_graficas_individuales.png"
    )
    fig.savefig(salida_individuales, dpi=150, bbox_inches="tight")
    print(f"Graficas individuales guardadas en: {salida_individuales}")
    plt.show()


if __name__ == "__main__":
    main()
