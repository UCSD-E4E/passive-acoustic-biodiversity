#!/bin/bash

#Program that leverages Sound Exchange (SoX) to create augmented wav file data
#1. Loops through a desired folder with wav files
#2. Creates an alternate version of each file with increased and decreased speed
#3. Creates an alternate version of each file with increased and decreased tempo
#SoX v14.4.1

echo "Speed and Tempo Perturbation Begin"

for sound_clip in *.WAV
do
	echo "Creating 1.1x speed version of $sound_clip"
	sox $sound_clip "${sound_clip%%.*}_SpeedUp.WAV" speed 1.1

	echo "Creating 0.9x speed version of $sound_clip"
        sox $sound_clip "${sound_clip%%.*}_SpeedDown.WAV" speed 0.9

	echo "Creating 1.1x tempo version of $sound_clip"
        sox $sound_clip "${sound_clip%%.*}_TempoUp.WAV" tempo 1.1

	echo "Creating 0.9x speed version of $sound_clip"
        sox $sound_clip "${sound_clip%%.*}_TempoDown.WAV" tempo 0.9

done

echo "Speed and Tempo Perturbation End"
