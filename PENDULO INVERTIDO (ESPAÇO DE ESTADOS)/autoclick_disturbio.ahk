#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

Esc::
Toggle := !Toggle
Loop
{
	If (!Toggle)
		Break
	sendinput {space}
	Sleep 86 ; Make this number higher for slower clicks, lower for faster 1000/(clicks/s).
}