"""
Ejercicio 2 - Optimizacion Multivariable


Objetivo: minimizar f(x, y) = x^2 + y^2 usando un cromosoma que
codifica dos variables en la misma cadena binaria.

Codificacion del cromosoma
Se usa un cromosoma de 8 bits: los primeros 4 bits codifican x y los
ultimos 4 bits codifican  y esto lo logramos
reutilizando `decodificar_cromosoma(cromosoma, rangos)` del motor base
que internamente divide el cromosoma en tantas partes iguales como
rangos se entreguen.

Minimizacion mediante un motor que maximiza

El motor `ejecutar_ag` siempre MAXIMIZA la funcion de aptitud que se le
entrega. Como f(x, y) = x^2 + y^2 tiene su minimo en (0, 0) con valor 0,
minimizarla es equivalente a maximizar su negativo: -f(x, y). Por eso
`funcion_aptitud` retorna el valor negativo, como sugiere el enunciado.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from genetic_algorithm import ejecutar_ag  # noqa: E402


# Espacio de busqueda para cada variable
RANGO_X = (-5.0, 5.0)
RANGO_Y = (-5.0, 5.0)

# 8 bits totales -> 4 bits para x, 4 bits para y
N_BITS_TOTAL = 8


def f(x: float, y: float) -> float:
    """Funcion objetivo a minimizar: f(x, y) = x^2 + y^2."""
    return x**2 + y**2


def funcion_aptitud(variables: list[float]) -> float:
    """Aptitud a MAXIMIZAR = -f(x, y), para minimizar f indirectamente."""
    x, y = variables
    return -f(x, y)


def main() -> None:
    resultado = ejecutar_ag(
        funcion_aptitud=funcion_aptitud,
        rangos=[RANGO_X, RANGO_Y],
        n_bits=N_BITS_TOTAL,
        tam_poblacion=60,
        n_generaciones=100,
        pc=0.8,
        pm=0.02,
        n_elite=1,
        semilla=42,
    )

    x_optimo, y_optimo = resultado.mejor_valor_decodificado
    valor_f = f(x_optimo, y_optimo)

    print("=== Ejercicio 2 - Optimizacion Multivariable ===")
    print(f"Mejor (x, y) encontrado: ({x_optimo:.4f}, {y_optimo:.4f})")
    print(f"f(x, y) minimo encontrado: {valor_f:.5f}")
    print("Optimo teorico esperado: (x, y) = (0, 0), f(x, y) = 0")

    # Grafica de convergencia (se grafica -aptitud = f(x,y) para leerlo
    # directamente como el valor que se esta minimizando)
    historial_f = [-v for v in resultado.historial_mejor_aptitud]
    historial_f_promedio = [-v for v in resultado.historial_aptitud_promedio]

    plt.figure(figsize=(8, 5))
    plt.plot(historial_f, label="Mejor f(x, y) (minimo)")
    plt.plot(historial_f_promedio, label="f(x, y) promedio", linestyle="--")
    plt.title("Convergencia AG - Ejercicio 2 (Minimizacion x² + y²)")
    plt.xlabel("Generacion")
    plt.ylabel("f(x, y)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    salida = os.path.join(
        os.path.dirname(__file__), "..", "resultados", "ejercicio2_convergencia.png"
    )
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Grafica guardada en: {salida}")
    plt.show()


if __name__ == "__main__":
    main()
