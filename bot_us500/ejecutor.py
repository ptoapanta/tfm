# =============================================================
# ejecutor.py — Envio de ordenes al broker via MT5
#
# La estrategia opera SOLO en LONG sobre el US500 (nunca en corto),
# de acuerdo con la señal del notebook (ruptura alcista de Donchian
# en tendencia alcista). Cada orden lleva SL y TP calculados a partir
# del ATR de la vela de la señal. Las operaciones se registran en
# operaciones.csv para su posterior analisis.
# =============================================================

import MetaTrader5 as mt5
import csv
import os
import logging
from datetime import datetime
from config import cfg
from riesgo import calcular_sl_tp

log = logging.getLogger("ejecutor")

# Codigos de MT5 que indican que la orden se acepto correctamente.
RETCODES_OK = {
    mt5.TRADE_RETCODE_DONE,
    mt5.TRADE_RETCODE_PLACED,
    mt5.TRADE_RETCODE_DONE_PARTIAL,
}


def enviar_orden_buy(atr: float, max_intentos: int = 3):
    """
    Envia una orden BUY a mercado con SL y TP basados en el ATR.
    Reintenta hasta max_intentos veces, refrescando el precio en
    cada intento.

    Retorna el ticket (int) de la posicion abierta si tuvo exito,
    o None si la orden fallo tras todos los intentos.
    """
    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None or tick.ask == 0:
        log.error(f"Sin cotizacion para {cfg.symbol}, no se envia la orden")
        return None

    entry  = tick.ask
    sl, tp = calcular_sl_tp(entry, atr)

    # Verificaciones de coherencia antes de enviar.
    if not (sl < entry < tp):
        log.error(f"Precios incoherentes: SL={sl} entry={entry} TP={tp}")
        return None

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
        "comment":      "agente_us500_donchian",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = None
    for intento in range(1, max_intentos + 1):
        log.info(
            f"Enviando BUY | intento {intento}/{max_intentos} | "
            f"entry={entry} | SL={sl} | TP={tp} | lote={cfg.lote}"
        )
        result = mt5.order_send(request)

        if result is None:
            log.warning(f"order_send devolvio None (intento {intento})")
            continue

        if result.retcode in RETCODES_OK:
            log.info(
                f"ORDEN BUY EJECUTADA | ticket={result.order} | "
                f"entry={entry} | SL={sl} | TP={tp}"
            )
            _registrar_operacion(entry, sl, tp, atr, result.order)
            return result.order

        log.warning(
            f"Fallo intento {intento} | "
            f"retcode={result.retcode} | {result.comment}"
        )

        # Refrescar el precio para el siguiente intento.
        tick = mt5.symbol_info_tick(cfg.symbol)
        if tick:
            entry = tick.ask
            sl, tp = calcular_sl_tp(entry, atr)
            request["price"] = entry
            request["sl"]    = sl
            request["tp"]    = tp

    log.error(
        f"La orden BUY fallo tras {max_intentos} intentos "
        f"(ultimo retcode={result.retcode if result else 'None'})"
    )
    return None


def cerrar_posicion(ticket: int) -> bool:
    """
    Cierra manualmente una posicion abierta por el agente
    (por ejemplo, al alcanzar el maximo de tiempo permitido).
    Retorna True si se cerro correctamente.
    """
    posiciones = mt5.positions_get(ticket=ticket)
    if not posiciones:
        log.warning(f"No se encontro la posicion {ticket} para cerrar")
        return False

    pos  = posiciones[0]
    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None:
        return False

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       cfg.symbol,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL,   # cierre de un BUY = SELL
        "position":     pos.ticket,
        "price":        tick.bid,
        "deviation":    20,
        "magic":        cfg.magic_number,
        "comment":      "cierre_por_tiempo",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode in RETCODES_OK:
        log.info(f"Posicion {ticket} cerrada manualmente | profit={pos.profit}")
        return True

    retcode = result.retcode if result else "None"
    log.error(f"Error cerrando posicion {ticket} | retcode={retcode}")
    return False


def _registrar_operacion(entry: float, sl: float, tp: float,
                         atr: float, ticket: int) -> None:
    """
    Anade una fila a operaciones.csv por cada orden ejecutada.
    Crea el archivo con cabecera si aun no existe.
    """
    archivo_nuevo = not os.path.exists(cfg.operaciones_csv)

    with open(cfg.operaciones_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow([
                "timestamp", "symbol", "tipo",
                "entry", "sl", "tp", "atr", "lote", "ticket"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg.symbol, "BUY",
            entry, sl, tp, round(atr, 4),
            cfg.lote, ticket
        ])
