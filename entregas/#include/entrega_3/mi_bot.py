import json
import os
import numpy as np


class Guardian:
    def __init__(self):
        ruta_train = os.path.join(os.path.dirname(__file__), "train.jsonl")

        with open(ruta_train, encoding="utf-8") as archivo:
            filas = [json.loads(linea) for linea in archivo if linea.strip()]

        x = np.array([self._features(caso) for caso in filas], dtype=float)
        y = np.array([caso["etiqueta"] for caso in filas], dtype=float)

        self.media = x.mean(axis=0)
        self.desviacion = x.std(axis=0) + 1e-6
        x = (x - self.media) / self.desviacion

        self.pesos = np.zeros(x.shape[1] + 1)

        for _ in range(800):
            z = self.pesos[0] + x @ self.pesos[1:]
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))

            error = p - y
            self.pesos[0] -= 0.12 * error.mean()
            self.pesos[1:] -= 0.12 * (
                (x.T @ error) / len(y) + 0.003 * self.pesos[1:]
            )

    @staticmethod
    def _features(caso):
        senales = caso["senales"]
        oraciones = senales.get("oraciones", [])
        fragmentos = senales.get("fragmentos", [])

        def valores(elementos, campo, defecto):
            resultado = [
                float(elemento.get(campo) or 0.0)
                for elemento in elementos
            ]
            return resultado or [defecto]

        respaldo_fragmentos = valores(
            fragmentos, "respaldo_todas", 1.0
        )
        contradiccion_fragmentos = valores(
            fragmentos, "contradiccion_todas", 0.0
        )

        llm = [oracion.get("llm") or {} for oracion in oraciones]

        def confianza(veredicto):
            return max(
                [
                    float(elemento.get("confianza") or 0.0)
                    for elemento in llm
                    if elemento.get("veredicto") == veredicto
                ],
                default=0.0,
            )

        return [
            1.0 - min(respaldo_fragmentos),
            max(contradiccion_fragmentos),
            confianza("no_respaldada"),
            confianza("contradicha"),
            confianza("respaldada"),
            min(1.0, len(oraciones) / 5.0),
            min(1.0, len(fragmentos) / 10.0),
        ]

    def score(self, caso):
        x = np.array(self._features(caso), dtype=float)
        x = (x - self.media) / self.desviacion

        z = self.pesos[0] + x @ self.pesos[1:]
        probabilidad = float(
            1 / (1 + np.exp(-np.clip(z, -30, 30)))
        )

        # Calibración equilibrada, monotónica y sin depender del idioma.
        return probabilidad / (probabilidad + 0.180)
