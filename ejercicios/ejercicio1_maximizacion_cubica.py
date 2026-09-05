"""
Ejercicio 1 - Maximizacion Cubica


Objetivo: maximizar f(x) = x^3 - 4x^2 + 5x usando un Algoritmo Genetico
con codificacion binaria.

Analisis matematico previo

f'(x) = 3x^2 - 8x + 5 = 0  ->  x = 1  o  x = 5/3 (=1.667)
f''(x) = 6x - 8
En x = 1:      f''(1) = -2  < 0   -> maximo local, f(1) = 2
En x = 5/3:    f''(5/3) = 2 > 0   -> minimo local

Se eligio el espacio de busqueda x en [0, 1.5] porque dentro de ese
rango la funcion crece desde f(0)=0 hasta un unico maximo claro en
x=1 (f(1)=2) y luego decrece hasta f(1.5)=1.875. Asi el AG debe
encontrar un maximo local inequivoco, sin ambiguedad con otro punto
de igual valor fuera del rango (lo cual si ocurriria, por ejemplo,
en un rango mas amplio como [0, 2], donde f(1) = f(2) = 2).

La decodificacion binaria mapea el cromosoma completo (16 bits) a un
unico valor real x dentro de [0, 1.5] usando `decodificar_cromosoma`.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from genetic_algorithm import ejecutar_ag  # noqa: E402


# Espacio de busqueda donde f(x) tiene un maximo local claro
RANGO_X = (0.0, 1.5)


def f(x: float) -> float:
    """Funcion objetivo a maximizar: f(x) = x^3 - 4x^2 + 5x."""
    return x**3 - 4 * x**2 + 5 * x


def funcion_aptitud(variables: list[float]) -> float:
    """Adapta f(x) al formato esperado por el motor del AG."""
    (x,) = variables
    return f(x)


def main() -> None:
    resultado = ejecutar_ag(
        funcion_aptitud=funcion_aptitud,
        rangos=[RANGO_X],
        n_bits=16,
        tam_poblacion=50,
        n_generaciones=100,
        pc=0.8,
        pm=0.01,
        n_elite=1,
        semilla=42,
    )

    x_optimo = resultado.mejor_valor_decodificado[0]
    print("=== Ejercicio 1 - Maximizacion Cubica ===")
    print(f"Mejor x encontrado: {x_optimo:.5f}")
    print(f"f(x) maximo encontrado: {resultado.mejor_aptitud:.5f}")
    print("Optimo teorico esperado: x=1.0, f(x)=2.0")

    # Grafica de convergencia
    plt.figure(figsize=(8, 5))
    plt.plot(resultado.historial_mejor_aptitud, label="Mejor aptitud")
    plt.plot(
        resultado.historial_aptitud_promedio,
        label="Aptitud promedio",
        linestyle="--",
    )
    plt.axhline(2.0, color="red", linestyle=":", label="Optimo teorico (2.0)")
    plt.title("Convergencia AG - Ejercicio 1 (Maximizacion Cubica)")
    plt.xlabel("Generacion")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    salida = os.path.join(
        os.path.dirname(__file__), "..", "resultados", "ejercicio1_convergencia.png"
    )
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Grafica guardada en: {salida}")
    plt.show()


if __name__ == "__main__":
    main()
