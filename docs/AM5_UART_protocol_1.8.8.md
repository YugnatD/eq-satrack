# AM5 — UART Protocol and Functions Observed in Firmware 1.8.8

Working reverse-engineering document based on `ZWO Mount Serial Communication Protocol_v1.7.pdf`, `main_AM5_1.8.8.bin`, and Ghidra analysis of the ESP32-S3 application code.

## Conventions

Commands normally start with `:` and end with `#`. Parameters are shown between angle brackets. Read-command responses normally end with `#`.

`1`/`0` generally mean success/failure, except for `:MS#` and GOTO commands, where `0` means success. Error responses use the form `e<n>#`.

## 1. Connections

| Interface | Observed/documented details |
|---|---|
| Serial | 9600 baud according to protocol v1.7 |
| Wi-Fi through hand controller | `192.168.4.1:4030` according to protocol v1.7 |
| USB | CDC/TinyUSB support is present in firmware 1.8.8 |
| BLE | A GATT server is present in firmware 1.8.8 |

## 2. Public commands documented in v1.7

### Mount modes

| Command | Response | Function |
|---|---|---|
| `:AP#` | None | Selects equatorial/GEM mode. The PDF says a reboot is required. |
| `:AA#` | None | Selects alt-azimuth mode. The PDF says a reboot is required. |

### Date and time

| Command | Response / format | Function |
|---|---|---|
| `:SC<MM/DD/YY>#` | `1` or `0` | Sets the date. |
| `:GC#` | `<MM/DD/YY>#` | Gets the date. |
| `:SL<HH:MM:SS>#` | `1` or `0` | Sets local time. |
| `:GL#` | `<HH:MM:SS>#` | Gets local time. |
| `:GS#` | `<HH:MM:SS>#` | Gets sidereal time. |
| `:GH#` | `1` or `0` | Gets daylight-saving-time state. |
| `:SH<n>#` | `1` | Sets daylight-saving time (`n=0` or `1`). |

### Location and coordinates

| Command | Response / format | Function |
|---|---|---|
| `:SG<s><HH>#` | `1` or `0` | Sets the time zone in hours. |
| `:SG<s><HH:MM>#` | `1` or `0` | Sets the time zone in hours and minutes. |
| `:GG#` | `<sHH:MM>#` | Gets the time zone. |
| `:St<s><DD*MM:SS>#` | `1` or `0` | Sets latitude. |
| `:Gt#` | `<sDD*MM:SS>#` | Gets latitude. |
| `:Sg<s><DDD*MM:SS>#` | `1` or `0` | Sets longitude. |
| `:Gg#` | `<sDDD*MM:SS>#` | Gets longitude. |
| `:Gm#` | `E`, `W`, or `N` | Gets current GEM orientation. |

### Target coordinates and motion

| Command | Response / format | Function |
|---|---|---|
| `:Sr<HH:MM:SS>#` | `1` or `0` | Sets target RA. |
| `:Gr#` | `<HH:MM:SS>#` | Gets target RA. |
| `:Sd<s><DD:MM:SS>#` | `1`, `0`, or an error | Sets target declination. |
| `:Gd#` | `<sDD*MM:SS>#` | Gets target declination. |
| `:GR#` | `<HH:MM:SS>#` | Gets current RA. |
| `:GD#` | `<sDD*MM:SS>#` | Gets current declination. |
| `:GZ#` | `<DDD*MM:SS>#` | Gets azimuth. |
| `:GA#` | `<sDD*MM:SS>#` | Gets altitude. |
| `:MS#` | `0` or `e<n>#` | Starts GOTO toward the prepared target. `0` means success. |
| `:Q#` | None | Stops motion. |

### Slew speed

| Command | Response | Function |
|---|---|---|
| `:R<n>#` | None | Selects speed level `0..9`. |
| `:RG#` | None | Selects 0.5× sidereal speed. |
| `:RC#` | None | Selects 1× sidereal speed. |
| `:RM#` | None | Selects 720× sidereal speed. |
| `:RS#` | None | Selects 1440× sidereal speed. |
| `:Rv<nnnn.nn>#` | None | Selects a variable speed from 0.00 to 1440.00× sidereal. |
| `:Rvr<nnnn.nn>#` | None | Firmware extension: sets variable speed for RA only. |
| `:Rvd<nnnn.nn>#` | None | Firmware extension: sets variable speed for DEC only. |
| `:Me#` / `:Qe#` | None | Move / stop toward east. |
| `:Mw#` / `:Qw#` | None | Move / stop toward west. |
| `:Mn#` / `:Qn#` | None | Move / stop toward north. |
| `:Ms#` / `:Qs#` | None | Move / stop toward south. |

#### Axis-specific variable speeds (firmware 1.8.8)

The v1.7 PDF documents only `:Rv<nnnn.nn>#`, but the 1.8.8 parser has two additional forms:

```text
:Rvr<nnnn.nn>#    RA only
:Rvd<nnnn.nn>#    DEC only
```

The character immediately after `Rv` is treated as an axis selector. `r` calls the RA-specific speed setter; `d` calls the DEC-specific speed setter. With no selector, `:Rv<nnnn.nn>#` calls a routine that updates both axis speed variables. The two axis-specific setters store their values in different firmware objects, so they are not aliases for one shared storage location.

The complete public manual-motion path has now been recovered. `:Me#` and `:Mw#` read the RA value set by `Rvr`; `:Mn#` and `:Ms#` read the DEC value set by `Rvd`. The two motor-start routines and their speed calculations are separate. Both axes can therefore run at the same time with different variable speeds. When tracking is active, the RA calculation also combines the manual RA rate with the tracking rate and applies the GEM-orientation sign rules. This makes coordinate differences measured before and after a short live test unsuitable for directly estimating the requested manual rate.

Example:

```text
:Rvr0002.00#    set RA to 2.00× sidereal
:Rvd0000.50#    set DEC to 0.50× sidereal
:Me#            start RA eastward
:Mn#            start DEC northward
```

The `Rvr`/`Rvd` forms are not listed in the public PDF, but they are handled directly by the AM5 UART parser. The command names are therefore firmware extensions, not standard LX200 commands. Their independent storage and use by the corresponding public manual-motion commands are both confirmed statically.

### Tracking and guiding

| Command | Response / format | Function |
|---|---|---|
| `:TQ#` | None | Sidereal tracking. |
| `:TS#` | None | Solar tracking. |
| `:TL#` | None | Lunar tracking. |
| `:GT#` | `<n>#` | Gets tracking-rate type. |
| `:Te#` | `1` or `0` | Starts tracking. |
| `:Td#` | `1` or `0` | Stops tracking. |
| `:GAT#` | `0#`, `1#`, or `e<n>#` | Gets tracking state. |
| `:Mg<d><nnnn>#` | None | Directional guiding for `0..3000` ms. |
| `:Rg<0.nn>#` | None | Sets guide rate from 0.10 to 0.90. |
| `:Ggr#` | `<0.nn>#` | Gets guide rate. |
| `:STa<nnsnn>#` | `1` or `0` | Sets meridian-crossing behavior. |
| `:GTa#` | `<nnsnn>#` | Gets meridian-crossing behavior. |

### Synchronization, home, and status

| Command | Response / format | Function |
|---|---|---|
| `:CM#` | `N/A#` or `e<n>#` | Synchronizes position using target RA/DEC. |
| `:hC#` | None | Mechanical return to home/zero position. |
| `:GU#` | Status string ending in `#` | Gets compact status, including mode, home, stall, guiding, park, and rates. |
| `:hP#` | `1` or `0` | GOTO to the default PARK position; GEM mode only. |

#### Exact `GU` compact-status format

`GU` builds a variable optional-character prefix followed by a fixed 9-character hexadecimal/decimal tail and `#`:

```text
<optional flags><RAflags:02x><DECflags:02x><NS-count:02x><RA-rate-digit><DEC-rate-digit><state-digit>#
```

Optional characters are appended in this exact order when their conditions are true:

| Character | Condition/effect |
|---|---|
| `n` | Operational state is not `1`. |
| `N` | Operational state is `0` or `1`. Thus state `0` produces `nN`, state `1` produces `N`, and higher states produce only `n`. |
| `L` | Altitude/limit control is enabled (`GLC = 1`). |
| `H` | Both per-axis home/reference bits are set. |
| `G` / `Z` | GEM mode / alt-azimuth mode. Exactly one is always emitted. |
| `C` | ESP32-S3 internal temperature is at or below `15.0 °C`. |
| `S` / `s` | RA / DEC motor diagnostic-fault latch was set. `GU` clears the corresponding bit while reporting it. |
| `T` / `t` | RA / DEC guide-input or timed-guide activity is currently detected. |
| `M` | Meridian-limit state byte is nonzero. |

The fixed tail then exposes the full RA and DEC flag bytes in two-digit lowercase hexadecimal, the active `NS` model-record count in two-digit hexadecimal, the current RA/DEC discrete rate-bucket digits (`0..9`), and the operational-state digit. Motor-timeout bit `0x40` is included in the raw flag byte once and then cleared after formatting. See selector `GFRa`/`GFDa` below for the axis-flag bit map.

The final state digit selects the firmware's six command-dispatch tables:

| State | Meaning |
|---:|---|
| `0` | Stopped/idle |
| `1` | Tracking |
| `2` | Coordinate GOTO/slew |
| `3` | Manual directional motion |
| `4` | Mechanical homing |
| `5` | Parking |

For example, the observed `NGM011000211#` separates as `N G M | 01 | 10 | 00 | 2 | 1 | 1`: GEM mode, meridian state nonzero, RA flags `0x01`, DEC flags `0x10`, no active `NS` records, rate buckets 2 and 1, operational state 1.

### Compound commands

| Command | Response / format | Function |
|---|---|---|
| `:SMGE<sDD*MM:SS>&<sDDD*MM:SS>#` | `1` or `0` | Sets latitude and longitude. |
| `:GMGE#` | `<latitude>&<longitude>#` | Gets latitude and longitude. |
| `:SMTI<MM/DD/YY>&<HH:MM:SS>&<sHH:MM>#` | `1` or `0` | Sets date, time, and time zone. |
| `:GMTI#` | `<date>&<time>&<time-zone>#` | Gets date, time, and time zone. |
| `:GMeq#` | `<target RA>&<target DEC>#` | Gets target RA and DEC. |
| `:GMEQ#` | `<current RA>&<current DEC>#` | Gets current RA and DEC. |
| `:GMZA#` | `<azimuth>&<altitude>#` | Gets azimuth and altitude. |
| `:SMeq<RA>&<DEC>#` | `0` or `e<n>#` | Sets RA/DEC and starts GOTO. |
| `:SMMC<RA>&<DEC>#` | `N/A#` or `e<n>#` | Sets RA/DEC and synchronizes. |

## 3. Undocumented commands confirmed in firmware 1.8.8

These commands are present in the main parser or in its secondary-controller path. Their practical availability may depend on the communication interface and on the secondary motor controller.

| Command / family | Expected response | Recovered function |
|---|---|---|
| `:GSN#` | No synchronous local reply | Forwards `:GSN#` to the secondary controller on internal channel `0x09`; intended as its serial-number query. Channel `0x09` replies enter the debug/log path rather than the normal USB response path. |
| `:GVE#` | No synchronous local reply | Forwards `:GVE#` on internal channel `0x09`; intended as the secondary firmware-version query. |
| `:GPT#` | No synchronous local reply | Forwards `:GPT#` on internal channel `0x09`; intended as the secondary product-type query. |
| `:GKEY#`, `:GADC#`, `:GWF#`, `:GWEB#`, `:GCAL#` | Empty `#` when sent directly | Not direct local commands. These strings are secondary-controller payloads emitted by `FTGh...` commands. |
| `:SH#`, `:SLED#` | Empty `#` or ordinary parser fallback when sent directly | Secondary-controller payloads emitted by `FTShh...` / `FTShled...`; not equivalent to the public `SH<n>` daylight-saving setter. |
| `:FT<secondary-command>#` | Usually `1`, `0`, or none | Maintenance and secondary-controller pass-through. |
| `:FTGU#` | Conditional maintenance-status string ending in `#` | Reports active-axis direction/rate pairs plus stopped/home markers while the FT gate is enabled; exact layout is documented below. |
| `:Rvr<nnnn.nn>#` | None | Sets variable manual speed for RA only. |
| `:Rvd<nnnn.nn>#` | None | Sets variable manual speed for DEC only. |
| `:STR<0.1..1.9>#` | `1` or `0` | Installs a RAM-only custom RA tracking-rate multiplier that overrides sidereal/solar/lunar timing until reboot. |
| `:SPl<pin>#` / `:SPh<pin>#` | None | Raw GPIO-low/high production command for GPIO `1..45`; dangerous and not suitable for probing. |
| `:GFR<selector>#` | Decimal/hex/flag response depending on selector | Reads internal RA-axis status/telemetry fields. |
| `:GFD<selector>#` | Decimal/hex/flag response depending on selector | Reads internal DEC-axis status/telemetry fields. |
| `:GFRf<g|z>#` / `:GFDf<g|z>#` | Decimal byte | Reads a low-level axis diagnostic value for the selected motor domain. |
| `:GBu#`, `:GBL#`, `:GBl#`, `:GBC#`, `:GBm#` | Decimal value or string | Buzzer mode, configured backlash, wireless-name suffix, live DEC backlash-compensation offset, and BLE state respectively. |
| `:GVT#`, `:GVD#`, `:GVP#`, `:GVB#`, `:GV#` | String or decimal value | Compile time, compile date, product name, hardware/configuration variant, and firmware version. |
| `:GFR0#`…`:GFR9#`, `:GFRa#`…`:GFRm#`, `:GFRt#` | Decimal, hexadecimal, character, or diagnostic string | RA motor/controller diagnostic fields. Not every letter is implemented; see the selector table below. |
| `:GFD0#`…`:GFD9#`, `:GFDa#`…`:GFDm#`, `:GFDt#` | Decimal, hexadecimal, character, or diagnostic string | DEC motor/controller diagnostic fields. Not every letter is implemented; see the selector table below. |
| `:GFRf<g|z>#`…`:GFRi<g|z>#` and DEC equivalents | Decimal byte | Four low-level getter combinations for two internal motor domains. |
| `:GMCR#`, `:GMCD#` | Decimal `0..31` | TMC2240 `DRV_STATUS.CS_ACTUAL` for RA and DEC: the current-scaling value actually used by the driver. |
| `:GMCr#`, `:GMCd#` | Decimal `10..31` | Reserved RAM parameters with compiled defaults `20` (RA) and `14` (DEC). Each has only its parser getter and setter in the complete 1.8.8 application cross-reference graph: neither value is consumed by motor control, written to NVS, or loaded from NVS. They must not be confused with `CS_ACTUAL`. |
| `:GMSR#`, `:GMSD#` | Decimal `0..127` | RA/DEC TMC2240 StallGuard2 threshold, encoded as `SGT + 64`; `64` therefore means signed `SGT = 0`. |
| `:GRl#`, `:GRT#`, `:GRR#` | Decimal value | Maximum manual rate, reduction ratio, and ESP-IDF reset-reason enum respectively. |
| `:GRr#`, `:GDr#` | RA / DEC coordinate | Aliases of `:GR#` and `:GD#` in firmware 1.8.8; live responses were identical. |
| `:GOr#`, `:GOd#` | `%.2f#` | Persisted RA and DEC home offsets converted from motor counts to angular units. |
| `:X...#` | See the detailed breakdown below | ADC, axis-register access, and axis-configuration functions. |
| `:NS...#` | Variable | Manages a 100-entry alignment/model table; detailed below. |
| `:NC#` | `1` or `0` | **Erases the default ESP-IDF `nvs` partition (factory reset)** when the mount is in the idle state. Do not use for probing. |

### Exact parser behavior

The main parser explicitly recognizes these prefixes:

| Prefix | Observed behavior |
|---|---|
| `GS` + `N` | Sends the five-byte payload `:GSN#` to the secondary controller on channel `0x09`; returns no local response. |
| `GV` + `E` | Sends the five-byte payload `:GVE#` to the secondary controller on channel `0x09`; returns no local response. |
| `GP` + `T` | Sends the five-byte payload `:GPT#` to the secondary controller on channel `0x09`; returns no local response. |
| `FT...` | Calls the maintenance/secondary-controller dispatcher. |
| `NS...` | Operates on the alignment/model sample table. |
| `NC` | Calls the NVS partition erase routine for partition label `nvs`; accepted only in operational state 0. |
| `Xa` | Reads ADC1 channel 8 / ESP32-S3 GPIO9 and returns calibrated millivolts as a decimal integer followed by `#`. |
| `Xb` | Reads ADC1 channel 6 / ESP32-S3 GPIO7 and returns calibrated millivolts as a decimal integer followed by `#`. |
| `Xc` | No response | Selects motor profile 2 on RA and profile 1 on DEC, rewriting TMC2240 chopper-mode and `IHOLD_IRUN` fields. |
| `Xd` | No response | Byte-for-byte identical parser branch to `Xc`; it performs the same RA-profile-2 / DEC-profile-1 change. |
| `Xer<register>#` | Reads one register from the first axis device and returns `<8-digit hexadecimal value>#`. |
| `Xed<register>#` | Reads one register from the second axis device and returns `<8-digit hexadecimal value>#`. The parser treats any selector other than `r` as the second device; this document uses `d` as the canonical DEC spelling, consistent with every other RA/DEC selector family. |
| `Xfr<reg><sep><value>#` | Writes a 32-bit value to one register on the first axis device and returns a two-digit hexadecimal status `<xx>#`. |
| `Xfd<reg><sep><value>#` | Writes a 32-bit value to one register on the second axis device and returns a two-digit hexadecimal status `<xx>#`. |

### `NS` alignment/model table and `NC` factory reset

The `NS` handler owns a fixed table of 100 records. A record is 48 bytes: five consecutive IEEE-754 `double` values at offsets `0x00..0x27`, an active byte at offset `0x28`, and seven bytes of padding. The first double is the sidereal time in hours at which the sample was created—the same value returned by `:GS#`. The remaining four doubles are the two transformed pointing coordinates and the two correction terms used by the model's spherical-coordinate interpolation. Records are created through the alternate synchronization callback installed by `:SSM1#`; ordinary synchronization is restored, and the table is erased, when `SSM` is set to another value.

`NSd` serializes the five doubles exactly as `%06.1f:%06.1f:%06.1f:%06.1f:%06.1f#`. The width `6` is a minimum width, not a hard length: a negative value or a value needing more digits can make a field longer. The index is a zero-based ordinal among active records, not necessarily the raw physical slot in the 100-entry array.

| Command | Response | Behavior |
|---|---|---|
| `:NS#` | One separately transmitted record per active entry; `0` if the table is empty | Exports every active record to the originating USB/network transport using the same five-field record formatter. This is a streaming/list operation, not a model-recompute command. The ordinary parser response is empty on success because the records are emitted directly by the export loop. |
| `:NST<aaaaaa><bbbbbb>#` | Exactly `<%06.2f><%06.2f>` with no delimiter and no terminating `#` | Copies exactly six bytes for each input field, parses each with the firmware's floating-point text parser, calls the model transform, and formats the two outputs independently with `%06.2f`. The width is a minimum, so the response can exceed 12 bytes for out-of-range values. On the live empty model, `:NST000000000000#` returned the 12 bytes `000.00000.00`. |
| `:NSC#` | `1` | Clears all 100 active flags and resets the record counters. |
| `:NSc#` | `<count>#` | Returns the active-record count. |
| `:NSd<index>#` | `%06.1f:%06.1f:%06.1f:%06.1f:%06.1f#` for a valid active ordinal; no reply for an invalid ordinal | Returns the indexed active record. The parser ignores the formatter's failure return, leaving the response empty when `index >= active_count` or `index >= 101`. Live confirmation with an empty table (`NSc -> 0#`) showed `NSd0` returning no bytes. |

`:NC#` is unrelated to the `NS` table despite the similar prefix. Its only idle-state branch calls the NVS flash erase wrapper with partition label `nvs`. It can erase location, limits, offsets, park points, BLE state, harmonic code, and other persisted calibration. It must be treated as a factory-reset command.

#### `X` command details

`Xa` and `Xb` use ADC unit 1 with 12-bit width and 11 dB attenuation. `Xa` reads channel 8 (GPIO9); `Xb` reads channel 6 (GPIO7). The raw sample is passed to the ESP-IDF `esp_adc_cal_raw_to_voltage`-equivalent routine, so the `%d#` result is calibrated **millivolts**. Five live samples gave `Xa = 853 mV` and `Xb = 2041..2043 mV`. Whole-program cross-reference analysis found no other application consumer of either ADC result, making these developer/production diagnostic probes rather than inputs to motion or safety logic. The firmware does not name the physical PCB nets connected to GPIO9 and GPIO7, so assigning those voltages to supply, current-sense, or another analog signal still requires PCB tracing or measurement.

`Xc` and `Xd` are identical and are not harmless queries. Both call the motor-profile routine as `(RA, profile 2)` followed by `(DEC, profile 1)`. Profile 1 sets TMC2240 `GCONF.en_pwm_mode` and profile 2 clears it, selecting the StealthChop/spreadCycle side of the driver configuration; the callback also rewrites the corresponding cached `CHOPCONF` value. A second callback writes profile-specific `IHOLD` and `IRUN` fields to TMC register `IHOLD_IRUN` (`0x10`).

For the tested GEM table, the live profile values were:

| Axis | Profile 1 (`IHOLD`, `IRUN`) | Profile 2 (`IHOLD`, `IRUN`) | State selected by `Xc` / `Xd` |
|---|---:|---:|---|
| RA | `8`, `15` | `8`, `28` | Profile 2 |
| DEC | `5`, `8` | `5`, `14` | Profile 1 |

The firmware contains separate GEM (`g`) and alt-azimuth (`z`) tables; they happened to contain the same values on the tested unit. Because `Xc`/`Xd` change chopper mode and motor current scaling, they were not exercised live.

`Xe` performs a direct TMC-style 32-bit register read through the selected axis device. The selector byte is `r` for the first device; every other selector is accepted and selects the second device. The numeric register address is parsed with `atoi()` and masked with `0x7f`, so the effective register address is always in the range `0x00..0x7f`; there is no explicit range or syntax validation in this handler.

The transaction contains an 8-bit register address and four data bytes. The handler calls the low-level read twice and returns the result of the second call. This is significant: the TMC2240 read protocol is pipelined, so a read request returns the requested register data on the following SPI datagram. The firmware’s double-read sequence is therefore consistent with a real TMC2240 register read, not an arbitrary retry.

The returned 32-bit value is formatted using `%08x#`, with no `0x` prefix and lowercase hexadecimal digits according to the C `%x` formatter.

#### Register-number encoding

The register text is parsed with C `atoi()`. It is therefore **decimal text**, not hexadecimal text. The response is hexadecimal, but the request is decimal. For example:

| TMC2240 register | Datasheet address | Decimal address to send | Read request |
|---|---:|---:|---|
| `GSTAT` | `0x01` | `1` | `:Xer1#` |
| `IFCNT` | `0x02` | `2` | `:Xer2#` |
| `NODECONF` | `0x03` | `3` | `:Xer3#` |
| `IOIN` | `0x04` | `4` | `:Xer4#` |
| `XACTUAL` | `0x21` | `33` | `:Xer33#` |
| `VACTUAL` | `0x22` | `34` | `:Xer34#` |
| `CHOPCONF` | `0x6c` | `108` | `:Xer108#` |
| `DRV_STATUS` | `0x6f` | `111` | `:Xer111#` |

The same decimal register-number rule applies to `Xed`. The firmware masks the parsed value with `0x7f`, so only the low seven address bits reach the TMC transaction. It does not validate that the requested address exists in the chip’s register map.

Canonical axis-named forms are:

```text
:Xer<register>#
:Xed<register>#
```

`Xf` performs a direct TMC-style register write. The selector is handled in the same way as `Xe`. The parser forcibly writes a NUL byte at `param[4]`, parses the register with `atoi(param+1)`, and parses the value with `atoi(param+5)`. Consequently, the parser expects three register characters at positions 1–3, one separator byte at position 4, and the decimal value beginning at position 5. The separator is not checked.

The effective layout after the `Xf` prefix is:

```text
selector | register digit 1 | register digit 2 | register digit 3 | ignored byte | decimal value
   [0]              [1]                  [2]                  [3]          [4]          [5...]
```

The value is also parsed by `atoi()`, so it is decimal text rather than hexadecimal text. The three-character register field is a consequence of the fixed parser offsets, not evidence that TMC register addresses are intrinsically three digits. A comma is conventional, for example `:Xfr033,123456#`; the comma is simply the ignored byte in this implementation.

The register address is written with bit `0x80` set. The value is transmitted as a 32-bit big-endian quantity. This exactly matches the TMC2240 SPI convention: read addresses have bit 7 clear, write addresses have bit 7 set, and every register transfer carries 32 data bits. The handler formats the low-level result with `%02x#`; the firmware does not return the written 32-bit value through this command. The code proves that this is the low-level transaction return byte, but does not by itself prove whether every bit is the TMC2240 status byte or a wrapper/driver status value.

Canonical axis-named forms are:

```text
:Xfr<3-digit-register><separator><value>#
:Xfd<3-digit-register><separator><value>#
```

The separator is not validated by the parser; a comma is a reasonable interoperable choice, but any byte in that position is accepted by this handler. The selector `r` selects device 0. The code treats every other selector as device 1; this document uses `d` as the canonical DEC spelling because that is the convention used by the other axis-specific command families.

### TMC2240 register-level confirmation

The firmware contains the string `TMC_2240` and a dedicated GPIO/SPI component. The `Xe`/`Xf` implementation independently confirms the following TMC-compatible properties:

| Property | Firmware evidence | TMC2240 correspondence |
|---|---|---|
| Register address width | Read masks address with `0x7f` | 7-bit register address plus read/write bit |
| Read operation | Address is masked to `0x7f`; read bit remains clear | TMC read access |
| Write operation | Adds `0x80` after the register address is reduced to 8 bits | TMC write access |
| Data width | Four-byte transfer | 32-bit register data |
| Read timing | Two read transactions | TMC pipelined read response |
| Write data order | Explicit byte extraction from most-significant to least-significant byte | Big-endian data bytes in the TMC SPI datagram |
| Read response | `%08x#` | Firmware-level representation of the returned 32-bit register value |

This confirms that `Xe`/`Xf` are register access commands, not generic opaque “internal values”. Most `Xe` reads do not alter configuration, but “read” must not be treated as universally side-effect-free: the TMC2240 SPI status documentation states that its latched `reset_flag` and `driver_error` indications are cleared by reading `GSTAT` (`0x01`). The firmware issues two read datagrams per `Xe`, so `:Xer1#` / `:Xed1#` can acknowledge those latches even though they do not write a register. Status registers such as `IFCNT`, `IOIN`, `XACTUAL`, `VACTUAL`, `MSCNT`, `DRV_STATUS`, `PWM_SCALE`, and `PWM_AUTO` are the useful read-only diagnostics; configuration registers should still be interpreted against the TMC2240 map. `Xf` writes can change motor current, chopper mode, limits, diagnostics, or driver state.

The official TMC2240 datasheet specifies a 40-bit SPI datagram: one address byte followed by four data bytes; read data is returned by the following transaction, and the SPI interface uses mode 3. It also specifies that the address MSB selects read/write and that the device returns status bits with each transaction. This independently confirms the firmware interpretation of the double-read sequence and address bit. See the [official TMC2240 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/tmc2240_datasheet.pdf), especially the SPI and register-map sections.

The `S...` dispatcher also contains undocumented subfamilies, including `SB`, `Sg`, `SI`, `SJ`, `Si`, `SZ`, `SO`, `SP`, `Sp`, and several `ST` variants. Their recovered branches are listed below; many directly change motor-driver calibration, home/park state, persistent NVS data, or secondary-controller state and must not be treated as harmless public commands.

### Exact `GFR` / `GFD` axis-diagnostic selectors

`GFR` selects the RA state object and `GFD` selects the DEC state object. Their selector implementations are structurally identical except where noted.

| Selector | Response format | Exact firmware source or operation |
|---|---|---|
| `0` | `%d#` | Timed-guide-active byte. The `Mg...` guide handlers set it when a pulse is accepted. |
| `1` | `%lld#` | Current STEP-pulse timer interval in microseconds. The common idle/1× value is about `22438`: `86,164.09 s / 3,840,000 STEP pulses = 22,438.6 µs` per pulse. During integer-rate acceleration/deceleration the firmware computes this interval by dividing the base interval by selector `5`. Its coordinate scale is `7,680,000` internal position counts per axis revolution, i.e. two coordinate counts per physical STEP pulse. The interval is independently stored for each axis and is passed to the per-axis timer-update routine. |
| `2` | `%ld#` | Current signed motor position in internal motor-count units. During live tracking, RA changed continuously while inactive DEC remained fixed. |
| `3` | `%.2f#` | Per-axis variable manual speed in sidereal multiples: the RA getter paired with `Rvr`, or DEC getter paired with `Rvd`. |
| `4` | `0#` or `1#` | Bit 0 of the flags byte at offset `0x06`; motor-active flag in the recovered motor-control code. |
| `5` | `%d#` | Current integer slew-rate scalar, in approximately sidereal-rate multiples. Acceleration/deceleration code ramps it one integer per control update toward the requested magnitude and recomputes selector `1` as base STEP interval divided by this value. Fractional/very-low rates use the minimum scalar `1`; motor-fault checking is suppressed below `1400`. |
| `6` | `%d#` | Reserved signed 32-bit field at offset `0x38`. Whole-program cross-reference analysis found no writer or consumer outside initialization and this diagnostic getter in 1.8.8; it remained `0` on both live axes. |
| `7` | `%d#` | Active TMC motor profile number (`1` or `2` in normal recovered paths), selecting chopper mode and the matching `IHOLD`/`IRUN` pair. |
| `8` | `%d#` | Bit 1 of the flags byte: pending stop/direction-reversal state. |
| `9` | `%ld#` | Signed GOTO/park target error in motor counts (`current position - target position`); normally zero outside those operations. |
| `a` | `%d#` | Complete flags byte at offset `0x06`; bit-level map follows this table. |
| `b` | `<character>#` | Direction byte at offset `0x04`: normally `e`/`w` for RA or `n`/`s` for DEC. |
| `c` | `%.2f#` | Per-axis floating motion-rate term in motor-count space. It is initialized to zero and written when a GOTO is prepared from the computed axis trajectory; the step-period calculation consumes it, and ST4/timed-guide code temporarily adds or subtracts the guide-rate term before that calculation. It is not the user variable speed—that is selector `3`. |
| `d` | `%d#` | Bit 2 of the flags byte: per-axis home/reference-acquired flag. Both axis bits produce `H` in the compact status. |
| `e` | `%x#` | Low nine bits of the current consecutive motor-fault diagnostic-sample counter, formatted in hexadecimal. The full counter increments when the selected fault GPIO remains low after at least 25 control iterations and only while rate field `5` is at least `1400`; it resets when those preconditions fail or a direction reversal is pending. A fault is raised when the full counter reaches `GSD`. Because the getter masks with `0x1ff`, displayed values wrap every 512 while `GSD` itself may be as high as 10000. |
| `f<g|z>` | `%d#` | Profile-1 TMC2240 `IRUN` for this axis. Final `g` selects the GEM table; `z` selects alt-azimuth. |
| `g<g|z>` | `%d#` | Profile-1 TMC2240 `IHOLD` for this axis, from the GEM or alt-azimuth table. |
| `h<g|z>` | `%d#` | Profile-2 TMC2240 `IRUN` for this axis, from the GEM or alt-azimuth table. |
| `i<g|z>` | `%d#` | Profile-2 TMC2240 `IHOLD` for this axis, from the GEM or alt-azimuth table. |
| `j` | `%d#` | TMC2240 `COOLCONF.SGT` StallGuard2 threshold for this axis, encoded as signed `SGT + 64`. This is an alias of `GMSR` for RA or `GMSD` for DEC. |
| `t` | `%d#` | Calls virtual callback slot 14. That slot is null for hardware variants `GVB=0..3`, including the tested AM5 (`GVB=1`), so the parser returns its initialized fallback `0`. It is not live telemetry on this AM5. |
| `k` | Label plus `%f#` | One-dimensional Kalman-filtered TMC2240 `SG_RESULT` load signal. RA: `kfp_out=<value>#`; DEC: `kfp_out = <value>#`. |
| `l` | Label plus `%f#` | Running maximum/reference of the filtered `SG_RESULT` signal. RA: `kfp_max = <value>#`; DEC: `g_kfp_max_axis2 = <value>#`. |
| `m` | `out:%f max%f#` | Current filtered `SG_RESULT` output and its running maximum/reference in one response. |

The letter selectors are case-sensitive. For example, `:GFRfg#` and `:GFRgg#` are different queries.

The `k/l/m` values are part of a second stall-detection path, not arbitrary debug floats. After the same active/rate/startup gates used by the motor monitor, raw per-axis `SG_RESULT` samples feed a scalar Kalman update with process-noise term `0.25`, measurement-noise term `0.95`, and initial estimate `1.0`. The firmware tracks the maximum filtered output, waits for 1000 samples, and evaluates the relative fall from that maximum. The DEC path treats a fall greater than 50% as a qualifying event; its separate trend counter must reach the hard-coded value `5` before setting the stall flag. This filtered trend detector is additional to the GPIO persistence counter exposed by selector `e` and configured by `SSD/GSD`.

The byte returned by selector `a` combines these per-axis flags:

| Bit | Mask | Recovered meaning |
|---:|---:|---|
| 0 | `0x01` | Motor/timer active. This is selector `4`. |
| 1 | `0x02` | Deceleration/target-stop or direction-reversal pending. This is selector `8`; fault sampling is inhibited while it is set. |
| 2 | `0x04` | Mechanical home/reference acquired for this axis. This is selector `d`; both axes set produces `H` in `GU`. |
| 3 | `0x08` | Axis participates in a sufficiently large GOTO correction. It is set when the prepared absolute target error exceeds 60 internal position counts and cleared when the residual error falls below 8 counts. |
| 4 | `0x10` | Per-axis stopped/move-complete latch. New motor-start paths clear it; target-error accumulation is skipped while it is set; state-transition code consumes and clears it after both motors stop. An inactive axis may therefore retain `0x10` while the other axis is tracking or moving. |
| 5 | `0x20` | Motor diagnostic/stall fault event. The fault monitor sets it, and `GU` emits `S` for RA or `s` for DEC and then clears the bit. |
| 6 | `0x40` | Motor-active timeout event. A watchdog stops the affected axis, sets this bit, and `GU` clears it after serializing the raw flag byte. |
| 7 | `0x80` | Raw sampled per-axis reference/limit input: ESP32-S3 GPIO38 for RA and GPIO4 for DEC, copied into the flags byte by the periodic input sampler. |

Selector `a` is formatted as an unsigned decimal byte even though several of its bits are transient. Reading `GU` has side effects on bits 5 and 6, so a `GFRa`/`GFDa` sample taken after `GU` may no longer contain those event latches.

Live read-only examples from the connected AM5 were:

```text
:GFRfg# -> 15#  :GFRgg# -> 8#  :GFRhg# -> 28#  :GFRig# -> 8#
:GFDfg# ->  8#  :GFDgg# -> 5#  :GFDhg# -> 14#  :GFDig# -> 5#
:GFRj#  -> 64#                :GFDj#  -> 64#
:GFRt#  -> 0#                 :GFDt#  -> 0#
:GFRk#  -> kfp_out=0.010000#  :GFDk# -> kfp_out = 0.010000#
```

Here `j = 64` decodes to signed `SGT = 0`. The `t = 0` results are unimplemented-callback fallbacks for this hardware variant, not measured values. Other numeric values are live state, not protocol constants.

The register meanings above follow the firmware's exact masks and the [official TMC2240 register map](https://www.analog.com/media/en/technical-documentation/data-sheets/tmc2240_datasheet.pdf): `IHOLD_IRUN` (`0x10`) contains five-bit `IHOLD` and `IRUN`, `DRV_STATUS` (`0x6F`) contains `SG_RESULT[9:0]` and `CS_ACTUAL[20:16]`, `COOLCONF` (`0x6D`) contains signed `SGT[22:16]`, and `TCOOLTHRS` is register `0x14`.

### Recovered persistent-parameter commands

The following names are supported by both parser behavior and the NVS key passed to the persistence routine.

| Get | Set | Accepted set value | Persistent key / function |
|---|---|---|---|
| `:GBL#` | `:SBL<n>#` | Integer `0..60` | `backlash`; setter is accepted only in the required GEM/state context. |
| `:GBm#` | `:SBm<0|1>#` | Boolean | `ble_state`; changing it starts or stops the corresponding BLE task. |
| `:GLH#` | `:SLH<n>#` | Integer `60..90` | `limit_h`. |
| `:GLL#` | `:SLL<n>#` | Integer `0..30` | `limit_l`. |
| `:GLC#` | `:SLE#` / `:SLD#` | Enable / disable | `limit_e`. |
| `:GRT#` | `:SRT<n>#` | Exactly `100` or `120` | `redu_ratio`. |
| `:GPW#` | `:SPW<n>#` | Parser accepts `0..256`; runtime values outside `79..256` are replaced by `79` | `pwm_duty`; RA brake-solenoid PWM target on ESP32-S3 GPIO40, detailed below. |
| `:GMhc#` | `:STh<string>#` | String copied by the parser | `harmoniccode`. |
| `:GTa#` | `:STa<n><n><sign><nn>#` | Validated structured value | Structured tracking/meridian-crossing configuration persisted under `track_limit`. |
| `:GML#` | `:SML<n>#` | Setter stores the low byte without range validation | Meridian-limit enable/state byte persisted under `meridian_limit`. The normal value is boolean-like (`0` or `1`), but the parser accepts any integer's low byte. |

`SBL`, `SLH`, `SLL`, `SLE`, `SLD`, `SRT`, `SPW`, `STh`, and `STa` mutate persistent configuration. They were not exercised during live validation.

#### `GPW` / `SPW`: RA brake-solenoid PWM

The PWM path can now be identified to high confidence as the normally-locked RA brake-solenoid drive, rather than an unspecified internal PWM value.

| Property | Recovered implementation |
|---|---|
| ESP32-S3 output | GPIO40 |
| Peripheral | LEDC low-speed mode, channel 0, timer 0 |
| Timer configuration | 100 kHz, 8-bit duty resolution |
| Power-on value | Raw duty `256` is requested for hardware variants other than zero |
| Stored/steady target | `GPW`; default/live value on the tested AM5 was `79` |
| Ramp | Every 100 ms, actual duty moves by `3` toward the target; the final step is clamped exactly to the target |
| Setter result | `SPW<n>` returns `1` for parsed values `0..256`, otherwise `0`; accepted values below `79` become `79` in the runtime target |

The main control task waits through its GPIO power-up sequence, requests duty `256`, and then runs the ramp controller every 100 ms. Reaching the live target `79` from `256` takes 59 updates, approximately 5.9 seconds. The local teardown report describes a spring-loaded, power-released brake only on RA, a power-on/off click, a 23-ohm solenoid, and about 3.7 V while energized. With a 12 V mount supply, `79 / 256 × 12 V = 3.70 V`, matching that measurement. The startup high duty followed by reduced holding duty is also the expected way to release a solenoid strongly and then limit its holding current and heat.

The firmware does not contain the PCB net name, so “RA brake solenoid” is a hardware-correlated identification rather than a source-symbol name. The independent agreement among GPIO PWM behavior, startup timing, duty ratio, measured voltage, and the teardown's RA-only brake makes the identification substantially stronger than a command-name inference. Changing `SPW` can alter brake release/holding force and coil heating; it must be treated as a safety-critical calibration, not a brightness or generic accessory control.

Additional `SB` controls recovered statically:

| Command | Response | Static behavior |
|---|---|---|
| `:SBu<n>#` | `1` for `n=0..2`, otherwise `0` | Selects a buzzer mode and starts a 100 ms buzzer action. `:GBu#` returns the selected mode. |
| `:SBe#` | None | Starts/enables the buzzer output path. |
| `:SBd#` | None | Stops/disables the buzzer output path and drives its GPIO low. |
| `:SBl<name>#` | `1` or `0` | Accepts one to six alphanumeric characters, rebuilds a wireless-service name/configuration, and restarts networking components. `:GBl#` reads the stored suffix/string. |
| `:SBm<0|1>#` | `1` or `0` | Changes and persists BLE state, then starts or stops the BLE service task. |

### Other undocumented `S` commands

These commands mutate runtime or persistent state. Their parser forms are confirmed, but they were deliberately not sent to the connected mount.

| Command | Response | Static behavior and validation |
|---|---|---|
| `:SRl<n>#` | `1` or `0` | Sets `maxrate_limit`; only `720..1440` is accepted. `:GRl#` reads it. |
| `:STR<rate>#` | `1` or `0` | Sets a RAM-only custom RA tracking-rate multiplier, accepted inclusively from `0.1` to `1.9`. While tracking, a nonzero value overrides the normal sidereal/solar/lunar interval selection and computes `RA STEP interval = base sidereal interval / rate`. It has no getter, no NVS write, and no recovered zero/clear path; after acceptance it remains active until reboot in firmware 1.8.8. This is distinct from `SRT`, which changes the persisted mechanical reduction ratio. |
| `:SSG<0|1>#` | `1` or `0` | Sets a RAM boolean returned by `:GSG#`. Static cross-reference analysis finds no consumer outside this setter/getter pair in 1.8.8, so it is currently a reserved/developer flag with no recovered motor effect. |
| `:SSD<n>#` | `1` or `0` | Accepts exactly `5..10000`. Sets the number of consecutive low diagnostic-input samples required before the motor-control code raises an axis fault. Sampling is armed only after 25 control iterations, only while diagnostic rate field `GFR5`/`GFD5` is at least `1400`, and not while the direction-reversal flag is set. `:GFRe#`/`:GFDe#` exposes the counter's low nine bits. `:GSD#` reads the threshold; default is `5`. |
| `:SSM<n>#` | `1` | Alignment/model collection mode. Value `1` installs the alternate synchronization callback used to populate/apply the `NS` model table. Any other value restores normal synchronization and clears all 100 in-RAM `NS` records. `:GSM#` reads the mode byte. |
| `:SSER<n>#` / `:SSED<n>#` | `1` | Writes a signed integer without validation to one element of an otherwise unused two-entry RAM table selected by the current mount mode. `R` selects the RA table; every other axis character selects the DEC table (`D` is canonical). Compiled table values are RA `{70 GEM, 90 alt-az}` and DEC `{200 GEM, 250 alt-az}`. Complete pointer/call-graph analysis finds no reader, no getter, and no persistence path for either table in 1.8.8, so these are dead/reserved developer writes with no recovered motor effect. The handler ignores whether the value changed and always returns `1`. |
| `:SMCr<n>#` / `:SMCd<n>#` | `1` or `0` | Sets the reserved RA/DEC RAM parameters returned by `:GMCr#` / `:GMCd#`; only `10..31` is accepted. Compiled defaults are `20` and `14`. Complete application cross-references contain no consumer and no persistence path, so changing them has no recovered motor effect in 1.8.8. |
| `:SMSR<n>#` / `:SMSD<n>#` | No normal reply in this branch | Writes RA/DEC TMC2240 `COOLCONF.SGT`. Protocol encoding is `n = signed SGT + 64`, so useful encoded range is `0..127` for hardware `SGT=-64..+63`. The parser itself does not range-check and the callback masks to seven bits. `GMSR`/`GMSD` and `GFRj`/`GFDj` read it back. |
| `:SML<n>#` | `1` | Sets and persists the meridian-limit enable/state byte (`meridian_limit`); `:GML#` reads it. No parser range check is applied, although normal operation treats zero/nonzero as disabled/enabled. |
| `:SPS<direction>#` | `1` or `0` | Preferred-pier-side override used by GOTO/coordinate side selection. `E/e` forces the east-side choice, `W/w` forces west, and `N/n` clears the override and restores automatic selection. |
| `:SPl<pin>#` / `:SPh<pin>#` | No reply | Raw production/debug GPIO control. Parses decimal GPIO `1..45`; `l` drives it low and every non-`l` selector reaching this branch drives it high (`h` is the intended spelling). It directly calls `gpio_set_level()` with no ownership or safe-pin check. This can toggle motor STEP/DIR/enable lines, sensors, the brake path, or communication pins and must never be used for probing. `SPW` and `SPS` are intercepted before this raw branch. |
| `:SZ<n>#` | Echoes `<n>#` | Writes RA TMC2240 register `TCOOLTHRS` (`0x14`), masking through the driver's register field. This changes the velocity threshold at which StallGuard2/CoolStep-related operation is enabled and is not a harmless diagnostic command. |
| `:SOa#` | `1` or `0` | When the mount has a valid home/reference state, converts the current RA/DEC motor positions into persisted home offsets and zeroes the current counters. |
| `:SOf#` | `1` | Clears and persists both home offsets. |
| `:STh<string>#` | `set harmonic code= <string><TAB>#` | Stores the harmonic-drive code. |

Known application-owned GPIOs show why `SPl`/`SPh` is unsafe:

| GPIO | Recovered firmware role |
|---:|---|
| 1, 2 | Outputs driven by the FT production/maintenance pattern logic; `FTSmer1` alternately pulses them at 100 ms steps |
| 4 | DEC reference/limit input copied to axis flag bit 7 |
| 7 | ADC1 channel 6 used by `Xb` |
| 9 | ADC1 channel 8 used by `Xa` |
| 10 | Active-low motor diagnostic/fault input in the RA path and several hardware-variant DEC paths |
| 21, 42 | Main-task power-up sequencing outputs |
| 26 | Alternate DEC diagnostic/fault input for one hardware variant |
| 34, 37 | RA ST4 guide inputs; active-low and applied with opposite signs |
| 35, 36 | DEC ST4 guide inputs; active-low and applied with opposite signs |
| 38 | RA reference/limit input copied to axis flag bit 7 |
| 40 | RA brake-solenoid LEDC PWM output |

This is not a complete board pinout; it lists only pins whose role is directly visible in the recovered application call graph. Raw GPIO writes bypass each subsystem's sequencing and state tracking.

The four case-sensitive tuning families below directly rewrite the TMC current-profile table. The suffix selects the GEM (`g`) or alt-azimuth (`z`) table and RA (`r`) or DEC (`d`). The final decimal text is reduced to its low byte; there is no five-bit range validation before storage.

```text
:SI<gr|gd|zr|zd><decimal>#   profile 2: write IRUN
:SJ<gr|gd|zr|zd><decimal>#   profile 2: write IHOLD
:Si<gr|gd|zr|zd><decimal>#   profile 1: write IRUN
:Sh<gr|gd|zr|zd><decimal>#   profile 1: write IHOLD
```

They are paired with the read-only `GFR/GFD` selectors `f`, `g`, `h`, and `i` as shown above. They are calibration/developer commands that control TMC current scaling, not coordinate setters, and should not be changed without a known-good table backup.

### Park-point management (`Sp` / `Gp`)

Firmware 1.8.8 stores up to four named park points. Each record contains an integer ID, a name of at most 31 characters, and RA/DEC motor coordinates.

| Command | Response | Operation |
|---|---|---|
| `:Gp#` | `<signed-ra-motor-count>:<signed-dec-motor-count>#` | Reads the currently selected park target as the two raw motor-coordinate integers used by the parking motion path. |
| `:Gps#` | `<state>#` | Reads the persisted park-state value. |
| `:Gpa#` | `<count>:[<id>,<name>,<signed-ra-count>,<signed-dec-count>]...#` | Lists every persisted park record. The response starts with the decimal record count, followed by one colon-prefixed bracketed record for each entry. There is no delimiter after the final `]` except the terminating `#`. |
| `:Sp01#` | Bare decimal result `<code>#` | Adds the current raw RA/DEC motor position as a park point, if in GEM mode and referenced/homed. It is rejected with `9#` while either axis is active. The persistent table has a hard maximum of four records. |
| `:Sp02<id><name>#` | Bare decimal result `<code>#` | Renames an existing park-point ID. Canonical `id` is exactly two decimal characters and `name` is `1..31` bytes. The parser bounds the total tail length but computes the ID by subtracting `'0'` without first validating that both ID bytes are digits. |
| `:Sp03<id>#` | Bare decimal result `<code>#` | Replaces the identified record's RA/DEC values with the current raw motor positions. The parser requires at least two ID bytes but does not validate that they are digits. |
| `:Sp04<id>#` | Bare decimal result `<code>#` | Deletes the identified persistent park record and compacts the remaining array. It has the same two-byte, non-validated ID parsing as `Sp03`. |
| `:Sp05<RA><DEC>#` | Error/result text only | Incomplete/dead branch. It writes a NUL at fixed offset 10, invokes the ordinary RA parser on the text at offset 2, and conditionally invokes the DEC parser at offset 11, but never calls any park-point add/update routine. Its control flow returns `e2#` for normally successful parse paths and may partially modify the ordinary target-coordinate globals on malformed paths. It cannot create a park point and must not be used. |
| `:Spu#` | `0` or `1` | Clears the completed-park state and returns success as a boolean. |

The `Sp01..Sp04` result is formatted as a bare decimal followed by `#`, not as the ordinary `e<n>#` GOTO-error family. Result values are: `1` success, `2` duplicate coordinates, `3` table full, `4` wrong mount mode, `5` not referenced/homed, `6` invalid name/arguments, `7` ID not found, `8` malformed input, and `9` motor active. In operational states 2, 3, or 4, the outer dispatcher instead rejects this family with `e3#` before the park handler runs.

### Command-routing architecture

Firmware 1.8.8 does not implement the UART interface as one monolithic LX200 parser. The recovered path is:

```text
USB / Wi-Fi / BLE text input
        -> framing and interface-origin record
        -> local command queue
        -> state-dependent command dispatcher
        -> local ESP32-S3 handler or secondary-controller queue
        -> response routed back to the originating interface
```

The secondary-controller UART uses the exact binary frame below. Text commands sent through it remain colon/hash framed inside the data field.

```text
offset  size  field
0       2     magic: F0 A5
2       2     N, big-endian: byte count of channel + data
4       1     channel/message type
5       N-1   data
4+N     2     checksum, big-endian
```

The checksum is `sum(channel byte + every data byte) & 0xffff`. It excludes `F0 A5`, the length, and the checksum itself. Total frame size is therefore `N + 6` bytes. The receive task uses the big-endian length, reads `N + 2` body bytes, and passes channel, data, and checksum to its callback. Surprisingly, no checksum comparison was found in that receive path: the two checksum bytes are discarded by length arithmetic before command parsing, but are not validated there.

Known channel/message-type values include `0x09` for maintenance/debug text (used by all recovered `FTGh...` forwarding), `0x0B` for `goto_home`, `0x0C` for `MODE_GEM` / `MODE_AZM`, `0x0D` for `tracking` / `no_track`, and `0x65` for the five-byte main-firmware version heartbeat/event (`1.8.8` in this image) sent after the power-up sequence. These values are internal UART message classes, not public LX200 commands. Incoming colon/hash text on a non-debug channel is put into the ordinary command queue; its response is sent back with the same channel byte. Channel `0x09` input is logged as debug text instead.

The local transport also reserves `UP`, `OK`, and `NG` for firmware-update flow, `WF` for Wi-Fi transfer, and `BS`/`BC` for Bluetooth services. These tokens are transport-control messages, not hidden mount-motion commands.

The command dispatcher has six behavior tables selected by the current operational state. All six share the common parser containing `G`, `S`, `R`, `X`, `FT`, `NS`, and `NC`; they override movement, stop, tracking, home, and synchronization behavior. This is why the same command can be accepted, ignored, or return an error depending on whether the mount is idle, tracking, manually moving, performing a GOTO, parking, or homing.

### Secondary-controller forwarding

The forwarding queue record contains one channel byte, a 16-bit data length, and the data bytes. It accepts data lengths below `0x33`, i.e. `0..50` bytes. The UART send task adds the `F0 A5` header, big-endian length, and additive checksum described above. If the queue is unavailable or full, the routine returns an internal error state rather than a normal public-protocol response.

All currently recovered `FTGh...` / `FTSh...` pass-through commands use channel `0x09`. Replies received on that channel enter the firmware's debug/log path rather than the normal USB command-response path. This explains why several live FT probes produced no immediate USB reply even though the outbound secondary message is real.

### `FT` subcommands recovered from the maintenance dispatcher

The `FT` branch is more structured than a single opaque pass-through. The following forms are explicitly compared by the firmware:

| Form | Static behavior |
|---|---|
| `:FTen<0|1>#` | No local reply. Stores a RAM gate for the remaining `FT` dispatcher and forwards exactly `:SFT0#` or `:SFT1#` to secondary channel `0x09`. Zero disables; every nonzero parsed value enables. The gate is not persisted. |
| `:FTGRd#` | Returns the current signed RA motor position divided by the axis counts-per-degree scale, truncated to an integer, as `<degrees>#`. Available only while the FT gate is enabled. |
| `:FTGDd#` | DEC equivalent of `FTGRd`: signed integer motor-axis degrees followed by `#`. |
| `:FTGU#` | Builds `[<RA-direction><RA-rate-digit>][<DEC-direction><DEC-rate-digit>][S][H]#`. Each bracketed axis pair is present only while that axis is active; the direction is its raw `e/w` or `n/s` byte and the rate is the same `0..9` bucket logic used by status reporting. `S` is appended while the mount state is stopped/idle, and `H` when both home bits are set. |
| `:FTGhsn#` | Forwards the exact secondary payload `:GSN#`. |
| `:FTGhve#` | Forwards the exact secondary payload `:GVE#`. |
| `:FTGhpt#` | Forwards the exact secondary payload `:GPT#`. |
| `:FTGhk#` | Forwards the exact secondary payload `:GKEY#`. |
| `:FTGhh#` | Forwards the exact secondary payload `:GADC#`. |
| `:FTGhwf<parameter>#` | Builds and forwards `:GWF#<parameter>`. |
| `:FTGhwe#` | Forwards the exact secondary payload `:GWEB#`. |
| `:FTGhca#` | Forwards the exact secondary payload `:GCAL#`. |
| `:FTMe<1..180>#`, `:FTMw<1..180>#` | Maintenance RA move. Converts the requested integer degrees to motor counts, applies the east/west and GEM-orientation signs, and starts the RA maintenance trajectory only in idle/manual state. Returns `1` when accepted and `0` for an invalid range, direction, state, or trajectory setup. |
| `:FTMs<1..180>#`, `:FTMn<1..180>#` | DEC equivalent. Converts integer degrees to signed motor counts and starts the DEC maintenance trajectory only in idle/manual state. The dispatcher returns `1` in an allowed state and `0` otherwise; the lower callback itself rejects values outside `1..180`. |
| `:FTShh<parameter>#` | No local reply. Builds and forwards the byte sequence `:SH#<parameter>` on channel `0x09`; note that the parameter is after the embedded `#`. |
| `:FTShled<parameter>#` | No local reply. Builds and forwards `:SLED#<parameter>` on channel `0x09`. |
| `:FTSmer0#`, `:FTSmer1#` | No ordinary payload. This is not the persisted meridian-limit setting. While FT mode is enabled, value `1` switches a main-loop GPIO1/GPIO2 production pattern from the normal 500 ms routine to a four-state 100 ms sequence: `(1,0)`, `(0,0)`, `(0,1)`, `(0,0)`, repeat. Value `0` restores the normal routine. It is RAM-only and should be treated as a hardware production test. |

The local `FT` paths above are code-resolved. The contents of replies generated inside the secondary controller for forwarded channel-`0x09` payloads still require that controller's firmware or an internal-UART capture. The maintenance move and GPIO-pattern commands should not be tested on a loaded mount.

### Read-only validation on a connected AM5

The AM5 responded at 9600 baud to the following diagnostic queries. These tests were read-only; no movement, register write, speed-setting, or configuration command was sent.

| Query family | Confirmed response format | Firmware-level interpretation |
|---|---|---|
| `:GFR0#` … `:GFR9#` | Decimal or fixed-point value followed by `#` | RA timed-guide flag, STEP interval, position, variable speed, motor state, rate, reserved field, profile, reversal state, and target error; see the exact selector table. |
| `:GFRa#` … `:GFRe#` | Flag, character, fixed-point, or hexadecimal value followed by `#` | RA complete flags, direction, computed trajectory-rate term, home flag, and fault-sample counter. |
| `:GFD0#` … `:GFD9#` | Decimal or fixed-point value followed by `#` | DEC equivalents of the numeric RA diagnostic fields. |
| `:GFDa#` … `:GFDe#` | Flag, character, fixed-point, or hexadecimal value followed by `#` | DEC equivalents of the lettered RA diagnostic fields. |
| `:GFRf<g|z>#` … `:GFRi<g|z>#` and DEC equivalents | Decimal byte followed by `#` | Four low-level diagnostic-result combinations for two motor domains. |
| `:GBu#`, `:GBL#`, `:GBl#`, `:GBC#`, `:GBm#` | Decimal or string followed by `#` | Buzzer mode, configured backlash, wireless-name suffix, live DEC backlash-compensation coordinate offset, and BLE state. `GBC` is updated on DEC direction reversals and cleared when the compensation state is reset. |
| `:GVT#`, `:GVD#`, `:GVP#`, `:GVB#`, `:GV#` | Time, build date, product, decimal value, or version string | Compile time/date, product, hardware/configuration variant, and firmware version. |
| `:GRl#`, `:GRT#`, `:GOr#`, `:GOd#` | Decimal or fixed-point value followed by `#` | Maximum rate, reduction ratio, and persisted RA/DEC home offsets. |
| `:GMCR#`, `:GMCD#` | Decimal value followed by `#` | RA/DEC TMC2240 `DRV_STATUS.CS_ACTUAL`, extracted as `(register >> 16) & 0x1f`. |
| `:GMCr#`, `:GMCd#` | Decimal value followed by `#` | Reserved RAM parameters constrained to `10..31`; the live values `20` and `14` exactly match the compiled defaults. Complete application cross-references show only their getter/setter pairs, with no motor-control or NVS consumer in 1.8.8. |
| `:GMSR#`, `:GMSD#` | Decimal value followed by `#` | RA/DEC TMC2240 signed `COOLCONF.SGT`, returned with a `+64` protocol bias. |
| `:NSc#` | Decimal active-record count followed by `#` | Read-only count of active alignment/model records. The latest sample returned `0#`. |
| `:NST<6-byte field><6-byte field>#` | Two `%06.2f` fields concatenated, with no delimiter and no final `#` | Pure model-coordinate transform; `:NST000000000000#` returned exactly `000.00000.00` with the empty model. |
| `:NSd<index>#` | One five-double record or no bytes | With `NSc = 0`, `:NSd0#` returned no bytes, confirming the statically recovered invalid-index behavior. |

Example values from the latest read-only session were `:GV# -> 1.8.8#`, `:GVT# -> 14:01:29#`, `:GVD# -> Jan 25 2026#`, `:GVP# -> AM5#`, `:GFR1# -> 22438#`, `:GFD1# -> 22438#`, `:GFR3# -> 1.00#`, and `:GFD3# -> 0.50#`. These values are session/device state, not fixed protocol constants. The earlier observed `GFD3 = 90.00` was a previously selected DEC variable speed, not an angle or fixed default.

Additional read-only observations from the same connected mount:

| Query | Observed response | Interpretation |
|---|---|---|
| `:GSN#`, `:GVE#`, `:GPT#` | No immediate response | Confirmed static behavior: each is forwarded on secondary channel `0x09`, whose incoming data is logged as debug text instead of being returned through the normal USB request/response path. |
| `:GKEY#`, `:GADC#`, `:GWF#`, `:GWEB#` | `#` | Direct forms are not recognized by the local `G` dispatcher; the empty response is its unknown-query fallback. |
| `:GCAL#` and `:GCAL0#` | `01/01/00#` | This text accidentally matches the ordinary `GC` date command plus ignored suffix (`AL`/`AL0`). It is not proof of direct calibration-query support. Use the secondary payload through `:FTGhca#` instead. |
| `:Xer1#`, `:Xed1#` | `00000001#` | TMC2240 `GSTAT` read on RA and DEC. |
| `:Xer2#`, `:Xed2#` | `00000000#` | TMC2240 `IFCNT` read on RA and DEC. |
| `:Xer4#`, `:Xed4#` | `11000043#`, `11000041#` | TMC2240 `IOIN` read; axis-specific hardware status differs. |
| `:Xer106#`, `:Xed106#` | `0000007c#`, `00000291#` | TMC2240 `MSCNT` read; values are live motor state. |
| `:Xer108#`, `:Xed108#` | `11008004#`, `11008004#` | TMC2240 `CHOPCONF` read. |
| `:Xer111#`, `:Xed111#` | `000f0000#`, `81050000#` | TMC2240 `DRV_STATUS` read; values differ between axes. |
| `:Xer112#`, `:Xed112#` | `00000000#`, `00000000#` | TMC2240 `PWMCONF` read in the tested state. |

Additional read-only commands recovered from the `G` dispatcher and validated live:

| Query | Observed response | Firmware-level meaning |
|---|---|---|
| `:GAST#` | `01/01/00&01:52:28&+00:00#08:04:30&+89*00:38#NGM011000211#` | Aggregated date/time/time-zone, current equatorial coordinates, and compact status. The response deliberately contains three `#`-terminated records. |
| `:GRr#` / `:GDr#` | `08:04:30#` / `+89*00:38#` | Aliases of `GR` / `GD`; three consecutive paired reads were identical. |
| `:GLH#` | `90#` | High limit value. It is paired with setter `SLH`. |
| `:GLL#` | `0#` | Low limit value. It is paired with setter `SLL`. |
| `:GLC#` | `1#` | Limit-control enable/state. It is paired with `SLE` and `SLD`. |
| `:GMA#` | `f412fa609611#` | Twelve hexadecimal digits produced from the device MAC address. |
| `:GMB#` | `AM5_609611#` | BLE/device advertising name derived from product name plus the low MAC bytes. |
| `:GBC#` | `0#` | Current DEC backlash-compensation coordinate offset; zero when no reversal compensation is pending. |
| `:GML#` | `1#` | Persisted meridian-limit enable/state byte (`meridian_limit`), paired with `SML<n>`. A nonzero value participates in the meridian-stop decision and adds `M` to the compact status. |
| `:GMhc#` | `3AC6H17100002972#` | Persisted harmonic-drive code (`harmoniccode`). |
| `:GNS#` | `#` | Serializes the currently selected `NS` alignment/model record, if its index is `0..99`, as five colon-separated fixed-point values: `%06.1f:%06.1f:%06.1f:%06.1f:%06.1f`. Empty means that no valid record is currently selected. |
| `:GPW#` | `79#` | RA brake-solenoid PWM holding-duty target on GPIO40. At 12 V, `79/256` corresponds to about 3.70 V average, matching the teardown measurement. |
| `:Gp#` | `0:0#` | Currently selected park target's raw RA and DEC motor coordinates. |
| `:Gps#` | `0#` | Park subsystem state. |
| `:Gpa#` | `1:[0,default,0,-1920000]#` | One persisted park record: ID `0`, name `default`, RA motor coordinate `0`, DEC motor coordinate `-1920000`. |
| `:GRR#` | `1#` | ESP-IDF `esp_reset_reason()` value. `1` is `ESP_RST_POWERON`, matching the live power-cycle test. |
| `:GSM#` | `1#` | Alignment/model collection mode paired with `SSM<n>` and the `NS` table. |
| `:GSG#` | `1#` | Reserved RAM flag paired with `SSG0`/`SSG1`; no non-parser consumer exists in firmware 1.8.8. |
| `:GSgd#` | `0#` | DEC TMC2240 `DRV_STATUS.SG_RESULT` (`0..1023`), the raw StallGuard2 load measurement. |
| `:GSgr#` | `0#` | RA TMC2240 `DRV_STATUS.SG_RESULT` (`0..1023`), the raw StallGuard2 load measurement. |
| `:GSD#` | `5#` | Motor-fault diagnostic-input persistence/debounce count. `SSD<n>` accepts `5..10000`; default is `5`. |
| `:GMCR#` / `:GMCD#` | `15#` / `5#` | Live RA/DEC TMC2240 `DRV_STATUS.CS_ACTUAL` values during this read-only sample. |
| `:GMSR#` / `:GMSD#` | `64#` / `64#` | Encoded StallGuard2 thresholds; both decode to signed `SGT = 0`. |
| `:GTS#` | `38.010201#` | ESP32-S3 internal temperature-sensor reading in degrees Celsius. A dedicated task refreshes it every second; at or below `15.0 °C`, `GU` includes the optional `C` flag. |
| `:Gh#` | `0#` | Mechanical home/reference-valid flag used to authorize park-point capture and home-offset operations. |

`GRR` returns the ESP-IDF `esp_reset_reason_t` numeric value without translating it to text:

| Value | ESP-IDF reason |
|---:|---|
| `0` | `ESP_RST_UNKNOWN` |
| `1` | `ESP_RST_POWERON` |
| `2` | `ESP_RST_EXT` (not applicable to ESP32-S3 in the normal implementation) |
| `3` | `ESP_RST_SW` |
| `4` | `ESP_RST_PANIC` |
| `5` | `ESP_RST_INT_WDT` |
| `6` | `ESP_RST_TASK_WDT` |
| `7` | `ESP_RST_WDT` |
| `8` | `ESP_RST_DEEPSLEEP` |
| `9` | `ESP_RST_BROWNOUT` |
| `10` | `ESP_RST_SDIO` |
| `11` | `ESP_RST_USB` |
| `12` | `ESP_RST_JTAG` |
| `13` | `ESP_RST_EFUSE` |
| `14` | `ESP_RST_PWR_GLITCH` |
| `15` | `ESP_RST_CPU_LOCKUP` |

The firmware implementation matches the ESP32-S3 ESP-IDF reset-reason constructor and getter: it converts the ROM reset cause once during startup, stores the enum, and `GRR` returns that stored value. The live `1#` after a power cycle is therefore a direct confirmation, not a name-based guess.

`:GX#` follows a special internal echo/forwarding path that constructs `:GX` instead of a normal payload. It produced no immediate USB response in the live test and must not be documented as an ordinary query returning `:GX#`.

Except for documented read-to-clear status behavior such as `GSTAT`, `Xe` does not write register configuration. The returned values above must not be treated as constants because several registers are live status or motion-state registers. No further raw-register probing should include `GSTAT` when preserving its latched reset/error evidence matters.

## 4. System functionality outside protocol v1.7

- OTA update support using ESP-IDF partition and OTA APIs.
- USB CDC/TinyUSB communication and update path.
- BLE GATT server with read, write, notification, indication, and MTU handling.
- Persistent NVS parameters: GEM/AZ mode, RA/DEC home offsets, maximum rate, reduction ratio, PWM, BLE state, park points, harmonic code, and calibration settings.
- Internal states for tracking limits, meridian limits, park, backlash, home, RA/DEC stall, ST4 guiding, and sensors.

### Reimplementation-relevant findings

Ghidra analysis confirms:

- The application has separate `uart1_recv_task` and `uart1_send_task` tasks.
- A baud-rate table contains `9600`, consistent with the public protocol.
- The motor layer references a `TMC_2240` component, with separate RA and DEC mode settings.
- Persistent keys include `longitude`, `latitude`, `homeoff_axis1`, `homeoff_axis2`, `maxrate_limit`, `redu_ratio`, `pwm_duty`, `ble_state`, `park_points`, and `harmoniccode`.
- The public parser delegates several commands to a secondary controller rather than implementing all behavior locally on the ESP32-S3.

## 5. Documented error codes

| Code | Meaning |
|---:|---|
| `e1#` | Parameter out of range. |
| `e2#` | Parameter-format error. |
| `e3#` | GOTO forbidden while homing, slewing, or already doing GOTO. |
| `e4#` | Equipment is already moving. |
| `e5#` | Target below the horizon. |
| `e6#` | Target below the configured height limit. |
| `e7#` | Time/location not synchronized. |
| `e8#` | Meridian already passed during tracking. |

## 6. Confidence and remaining limits

- **Code-confirmed:** parser spelling/case, numeric parsing, response formatter, bounds checks, RAM/NVS destinations, motor-state fields, TMC register masks, secondary-UART framing, GPIO numbers, PWM configuration, and call graph described as exact above.
- **Live read-only confirmed:** ordinary `G...` replies, axis-diagnostic formats and representative values, ADC voltages, selected TMC register reads, reset reason, temperature, PWM target, home/model states, and independent RA/DEC variable-speed storage. Live values are examples rather than protocol constants.
- **Hardware-correlated:** GPIO40 PWM is identified as the RA brake-solenoid drive from its startup/full-to-hold sequence and the teardown's independent 23-ohm/3.7 V brake measurement. The binary contains no PCB net label, so this conclusion is stronger than a name guess but still depends on that external hardware observation.
- **Still requires PCB or secondary-firmware evidence:** the physical nets on ADC GPIO9/GPIO7 and the semantic contents of replies generated inside the secondary controller for `GSN`, `GVE`, `GPT`, and other channel-`0x09` maintenance payloads.
- **Deliberately not live-tested:** configuration setters, NVS erase, raw GPIO control, TMC writes, current/profile changes, home/park calibration, custom tracking-rate override, and other commands that can move hardware or alter safety-critical state.

Movement, calibration, limit, and update commands should not be tested on a loaded mount. A no-load test bench, current-limited supply, and UART capture are recommended.
