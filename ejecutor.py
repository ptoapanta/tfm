# =============================================================
# ejecutor.py — Envío de órdenes al broker via MT5
#
# SOLO opera en LONG (nunca en corto en el US500).
# Reintenta hasta 3 veces si la orden falla.
# Registra cada operación en operaciones.csv
# =============================================================

import MetaTrader5 as mt5
import csv
import os
import logging
from datetime import datetime
from config import cfg
from riesgo import calcular_precios_long

log = logging.getLogger("ejecutor")

# Códigos de MT5 que significan orden ejecutada correctamente
RETCODES_OK = {
    mt5.TRADE_RETCODE_DONE,
    mt5.TRADE_RETCODE_PLACED,
    mt5.TRADE_RETCODE_DONE_PARTIAL,
}


def enviar_orden_buy(max_intentos: int = 3) -> bool:
    """
    Envía una orden BUY al broker con SL y TP basados en ATR.
    Reintenta hasta max_intentos veces si falla.

    Retorna True si la orden se ejecutó, False si falló.
    """
    entry, sl, tp = calcular_precios_long()

    # Verificar que los precios son válidos
    if entry <= 0 or sl <= 0 or tp <= 0:
        log.error(f"Precios inválidos: entry={entry} sl={sl} tp={tp}")
        return False

    if sl >= entry:
        log.error(f"SL ({sl}) debe ser menor que entry ({entry})")
        return False

    if tp <= entry:
        log.error(f"TP ({tp}) debe ser mayor que entry ({entry})")
        return False

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       cfg.symbol,
        "volume":       cfg.lote,
        "type":         mt5.ORDER_TYPE_BUY,
        "price":        entry,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        cfg.magic_number,
        "comment":      "agente_us500_v3",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    for intento in range(1, max_intentos + 1):
        log.info(
            f"Enviando BUY | intento {intento}/{max_intentos} | "
            f"entry={entry} | SL={sl} | TP={tp} | lote={cfg.lote}"
        )

        result = mt5.order_send(request)

        if result is None:
            log.warning(f"order_send devolvió None — intento {intento}")
            continue

        if result.retcode in RETCODES_OK:
            log.info(
                f"ORDEN BUY OK | ticket={result.order} | "
                f"entry={entry} | SL={sl} | TP={tp}"
            )
            _registrar_operacion("BUY", entry, sl, tp, result.order)
            return True

        # Error recuperable — actualizar precio y reintentar
        log.warning(
            f"Fallo intento {intento} | "
            f"retcode={result.retcode} | {result.comment}"
        )

        # Actualizar precio para el siguiente intento
        tick = mt5.symbol_info_tick(cfg.symbol)
        if tick:
            request["price"] = tick.ask

    log.error(
        f"Orden BUY falló tras {max_intentos} intentos | "
        f"último retcode={result.retcode if result else 'None'}"
    )
    return False


def cerrar_posiciones_agente() -> int:
    """
    Cierra todas las posiciones abiertas por este agente
    identificadas por el magic_number.

    Retorna el número de posiciones cerradas correctamente.
    """
    posiciones = mt5.positions_get(symbol=cfg.symbol)
    if not posiciones:
        return 0

    cerradas = 0
    for pos in posiciones:
        if pos.magic != cfg.magic_number:
            continue

        tick    = mt5.symbol_info_tick(cfg.symbol)
        precio  = tick.bid  # cierre de BUY al precio BID

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       cfg.symbol,
            "volume":       pos.volume,
            "type":         mt5.ORDER_TYPE_SELL,
            "position":     pos.ticket,
            "price":        precio,
            "deviation":    20,
            "magic":        cfg.magic_number,
            "comment":      "cierre_agente",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode in RETCODES_OK:
            log.info(f"Posición {pos.ticket} cerrada | profit={pos.profit}")
            cerradas += 1
        else:
            retcode = result.retcode if result else "None"
            log.error(
                f"Error cerrando posición {pos.ticket} | "
                f"retcode={retcode}"
            )

    return cerradas


def _registrar_operacion(tipo: str, entry: float,
                          sl: float, tp: float, ticket: int) -> None:
    """
    Guarda cada operación ejecutada en operaciones.csv.
    Crea el archivo con cabecera si no existe.
    """
    archivo_nuevo = not os.path.exists(cfg.operaciones_csv)

    with open(cfg.operaciones_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow([
                "timestamp", "symbol", "tipo",
                "entry", "sl", "tp", "lote", "ticket"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg.symbol, tipo,
            entry, sl, tp,
            cfg.lote, ticket
        ])