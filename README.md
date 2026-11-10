# Enrutamiento adaptativo · Sink Tree y rutas óptimas

## Cómo ejecutarlo (Windows, macOS o Linux)

Necesitas Python 3.9 o superior.

```bash
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Se abre solo en el navegador (http://localhost:8501).

## Uso rápido

1. **Elige el destino**: en la barra lateral o haciendo clic sobre un nodo del grafo.
2. **Cambia el peso de un camino**: haz clic en el número que está sobre el camino
   (aparece un editor en la barra lateral) o edita la tabla en «Construir / editar red».
3. El sistema suma los pesos y marca las **3 rutas más óptimas**
   (verde la 1ª, naranja la 2ª y morada la 3ª). La tabla de abajo muestra la suma, por ejemplo `60 + 110 + 70 + 80 = 320`.
4. **Sink Tree**: la pestaña Sink Tree dibuja el árbol de rutas óptimas de todos los nodos hacia el destino.
5. **Tiempo real**: con la simulación activa, la carga, el retardo, el ancho de banda y las
   fallas cambian solos y las rutas se recalculan. El historial explica cada cambio de ruta.
6. Puedes agregar y quitar nodos y caminos, o generar una red aleatoria de 3 a 60 nodos.
   La red se guarda y se carga en JSON desde «Exportar».

## Archivos

- `main.py`: motor de cálculo (Dijkstra, Sink Tree, Yen, simulación) + interfaz Streamlit
- `requirements.txt`: librerías necesarias
