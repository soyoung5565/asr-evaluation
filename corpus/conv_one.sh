#!/bin/bash
f="$1"
b=$(basename "$f" .mp3)
out="/d/asr/corpus/cv26_enmy_wav/$b.wav"
[ -f "$out" ] && exit 0
/usr/bin/ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 "$out" 2>/dev/null || exit 0
