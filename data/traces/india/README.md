# Our own Indian network traces (U1)

Recorded with `tools/trace_logger.py`. Every trace file sits next to a `<name>.meta.json` with the network,
route and data used. Evaluate with `python evaluate.py --traces india --oracle`.

## Recording protocol

1. Tether a laptop to the phone hotspot (or run the script on Android in Termux: `pkg install python`).
2. Turn off other downloads/streams on the phone; keep the screen on.
3. Start recording **just before** the trip segment and name it `<city>_<mode>_<route>`:

   ```bash
   python tools/trace_logger.py --name delhi_metro_yellow_rajivchowk_hudacity \
       --duration 900 --network "Jio 4G" --notes "Rajiv Chowk -> HUDA City Centre, evening rush"
   ```

4. Aim for 5–10 minute traces (a video is ~3.2 min; longer traces loop) and **at least 10 traces** across
   different situations: metro (underground + elevated), bus/auto, intercity train, crowded market,
   campus Wi-Fi, home broadband.
5. Data usage: the default 8 Mbps cap uses at most ~60 MB/min (`--max-mb` is a hard stop, default 400 MB).

## Planned traces

| Name | Network | Situation | Status |
|---|---|---|---|
| | | | |
