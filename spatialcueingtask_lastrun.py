#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.0),
    on Mon Apr 21 16:42:23 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.0'
expName = 'spatial cueing task first version'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/asyasorgulu/Desktop/spatialcueing/spatialcueingtask_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 800], fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1, -1, -1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('continuekey') is None:
        # initialise continuekey
        continuekey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continuekey',
        )
    if deviceManager.getDevice('consentresp') is None:
        # initialise consentresp
        consentresp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='consentresp',
        )
    if deviceManager.getDevice('practicekey') is None:
        # initialise practicekey
        practicekey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practicekey',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('Startkey') is None:
        # initialise Startkey
        Startkey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Startkey',
        )
    if deviceManager.getDevice('key_resp2') is None:
        # initialise key_resp2
        key_resp2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruct" ---
    instructText = visual.TextStim(win=win, name='instructText',
        text='Welcome to the experiment!\n\nThis experiment aims provide insight into how our brain reacts to different type of emotional visual stimuli. \n\nYou will be asked to complete a task where you respond to visual targets appearing either on the left or right side of the screen. Before each target appears, a cue will be shown, which is either a neutral gaze or an emotional face with the eyes looking in a specific direction.\n\nYour task is to respond as quickly and accurately as possible when the target dot appears.\n\nYour participation is anonymous and you are free to withdraw at any time without giving a reason.\n\nFurther instructions will follow after the consent form.\n\n',
        font='Arial',
        units='height', pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    continuekey = keyboard.Keyboard(deviceName='continuekey')
    introkey = visual.TextBox2(
         win, text="PRESS 'SPACE' TO CONTINUE", placeholder='Type here...', font='Arial',
         pos=(0, -0.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=[0.3569, 0.6941, 0.8039], borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='introkey',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "consent" ---
    text = visual.TextStim(win=win, name='text',
        text='Consent Form\n\nBy pressing ‘Y’, you confirm that:\n\n- You are 18 years or older\n- You have normal or corrected-to-normal vision and hearing\n- You understand what the study involves\n- You are participating voluntarily and may withdraw at any time\n- You consent to your anonymous data being used for research purposes\n\nPress ‘Y’ to continue.\nPress ‘N’ if you do not consent – the experiment will end.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    consentresp = keyboard.Keyboard(deviceName='consentresp')
    
    # --- Initialize components for Routine "welcome" ---
    welcometext = visual.TextStim(win=win, name='welcometext',
        text='Welcome to the experiment!\n\nIn this task, you will see a face cue followed by a target and your goal is to respond as quickly and accurately as possible:\n\n- Press the "A" key if the target appears on the LEFT  \n- Press the "L" key if the target appears on the RIGHT\n\nYou will start with a short practice round  \nbefore the main experiment begins.\n\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start = visual.TextBox2(
         win, text="PRESS 'SPACE' TO START", placeholder='Type here...', font='Arial',
         pos=(0, -0.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=[0.3569, 0.6941, 0.8039], borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='start',
         depth=-1, autoLog=True,
    )
    practicekey = keyboard.Keyboard(deviceName='practicekey')
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='',
        font='Arial',
        units='height', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    cue = visual.ImageStim(
        win=win,
        name='cue', units='height', 
        image=None, mask=None, anchor='center',
        ori=1.0, pos=(0, 0), size=(.25, .25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    target = visual.ImageStim(
        win=win,
        name='target', units='height', 
        image='targetimage/circle_grey.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(.25, .25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from track_rt
    valid_rts = []
    invalid_rts = []
    reminder = visual.TextStim(win=win, name='reminder',
        text='Click on a key!',
        font='Arial',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    trial_counter = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, -0.45),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=0.8,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_counter',
         depth=-7, autoLog=True,
    )
    # Run 'Begin Experiment' code from fbcode
    
    
    
    # --- Initialize components for Routine "feedback1" ---
    feedbackTxt1 = visual.TextStim(win=win, name='feedbackTxt1',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trialClock = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, -0.45),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=0.8,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trialClock',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "endofpractice" ---
    instructText1 = visual.TextStim(win=win, name='instructText1',
        text='Practice is over — time for the real task!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Startkey = keyboard.Keyboard(deviceName='Startkey')
    start1 = visual.TextBox2(
         win, text="PRESS 'SPACE' TO START", placeholder='Type here...', font='Arial',
         pos=(0, -0.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=[0.3569, 0.6941, 0.8039], borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='start1',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "trial1" ---
    fixation1 = visual.TextStim(win=win, name='fixation1',
        text='+',
        font='Arial',
        units='height', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    cueimg = visual.ImageStim(
        win=win,
        name='cueimg', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=1.0, pos=(0, 0), size=(.25, .25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    target1 = visual.ImageStim(
        win=win,
        name='target1', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(.25, .25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-4.0)
    key_resp2 = keyboard.Keyboard(deviceName='key_resp2')
    # Run 'Begin Experiment' code from track_rt1
    valid_rts = []
    invalid_rts = []
    reminder1 = visual.TextStim(win=win, name='reminder1',
        text='Click a key to respond!',
        font='Arial',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "finish" ---
    finishMsg_2 = visual.TextStim(win=win, name='finishMsg_2',
        text=None,
        font='Arial',
        units='height', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from endingfb
    valid_rts = []
    invalid_rts = []
    
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instruct" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruct.started', globalClock.getTime(format='float'))
    continuekey.keys = []
    continuekey.rt = []
    _continuekey_allKeys = []
    introkey.reset()
    # keep track of which components have finished
    InstructComponents = [instructText, continuekey, introkey]
    for thisComponent in InstructComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruct" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructText* updates
        
        # if instructText is starting this frame...
        if instructText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructText.frameNStart = frameN  # exact frame index
            instructText.tStart = t  # local t and not account for scr refresh
            instructText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructText.started')
            # update status
            instructText.status = STARTED
            instructText.setAutoDraw(True)
        
        # if instructText is active this frame...
        if instructText.status == STARTED:
            # update params
            pass
        
        # *continuekey* updates
        waitOnFlip = False
        
        # if continuekey is starting this frame...
        if continuekey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continuekey.frameNStart = frameN  # exact frame index
            continuekey.tStart = t  # local t and not account for scr refresh
            continuekey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continuekey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continuekey.started')
            # update status
            continuekey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continuekey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continuekey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continuekey.status == STARTED and not waitOnFlip:
            theseKeys = continuekey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _continuekey_allKeys.extend(theseKeys)
            if len(_continuekey_allKeys):
                continuekey.keys = _continuekey_allKeys[-1].name  # just the last key pressed
                continuekey.rt = _continuekey_allKeys[-1].rt
                continuekey.duration = _continuekey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *introkey* updates
        
        # if introkey is starting this frame...
        if introkey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            introkey.frameNStart = frameN  # exact frame index
            introkey.tStart = t  # local t and not account for scr refresh
            introkey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(introkey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'introkey.started')
            # update status
            introkey.status = STARTED
            introkey.setAutoDraw(True)
        
        # if introkey is active this frame...
        if introkey.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruct" ---
    for thisComponent in InstructComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruct.stopped', globalClock.getTime(format='float'))
    # check responses
    if continuekey.keys in ['', [], None]:  # No response was made
        continuekey.keys = None
    thisExp.addData('continuekey.keys',continuekey.keys)
    if continuekey.keys != None:  # we had a response
        thisExp.addData('continuekey.rt', continuekey.rt)
        thisExp.addData('continuekey.duration', continuekey.duration)
    thisExp.nextEntry()
    # the Routine "Instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "consent" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('consent.started', globalClock.getTime(format='float'))
    consentresp.keys = []
    consentresp.rt = []
    _consentresp_allKeys = []
    # keep track of which components have finished
    consentComponents = [text, consentresp]
    for thisComponent in consentComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "consent" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *consentresp* updates
        waitOnFlip = False
        
        # if consentresp is starting this frame...
        if consentresp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            consentresp.frameNStart = frameN  # exact frame index
            consentresp.tStart = t  # local t and not account for scr refresh
            consentresp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(consentresp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'consentresp.started')
            # update status
            consentresp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(consentresp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(consentresp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if consentresp.status == STARTED and not waitOnFlip:
            theseKeys = consentresp.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
            _consentresp_allKeys.extend(theseKeys)
            if len(_consentresp_allKeys):
                consentresp.keys = _consentresp_allKeys[-1].name  # just the last key pressed
                consentresp.rt = _consentresp_allKeys[-1].rt
                consentresp.duration = _consentresp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from consentcode
        keys = event.getKeys()
        
        if 'y' in keys:
            continueRoutine = False  
        elif 'n' in keys:
            core.quit()  
        
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in consentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "consent" ---
    for thisComponent in consentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('consent.stopped', globalClock.getTime(format='float'))
    # check responses
    if consentresp.keys in ['', [], None]:  # No response was made
        consentresp.keys = None
    thisExp.addData('consentresp.keys',consentresp.keys)
    if consentresp.keys != None:  # we had a response
        thisExp.addData('consentresp.rt', consentresp.rt)
        thisExp.addData('consentresp.duration', consentresp.duration)
    # Run 'End Routine' code from consentcode
    
    
    
    
    
    thisExp.nextEntry()
    # the Routine "consent" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime(format='float'))
    start.reset()
    practicekey.keys = []
    practicekey.rt = []
    _practicekey_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [welcometext, start, practicekey]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcometext* updates
        
        # if welcometext is starting this frame...
        if welcometext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcometext.frameNStart = frameN  # exact frame index
            welcometext.tStart = t  # local t and not account for scr refresh
            welcometext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcometext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcometext.started')
            # update status
            welcometext.status = STARTED
            welcometext.setAutoDraw(True)
        
        # if welcometext is active this frame...
        if welcometext.status == STARTED:
            # update params
            pass
        
        # *start* updates
        
        # if start is starting this frame...
        if start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start.frameNStart = frameN  # exact frame index
            start.tStart = t  # local t and not account for scr refresh
            start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start.started')
            # update status
            start.status = STARTED
            start.setAutoDraw(True)
        
        # if start is active this frame...
        if start.status == STARTED:
            # update params
            pass
        
        # *practicekey* updates
        waitOnFlip = False
        
        # if practicekey is starting this frame...
        if practicekey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practicekey.frameNStart = frameN  # exact frame index
            practicekey.tStart = t  # local t and not account for scr refresh
            practicekey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practicekey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practicekey.started')
            # update status
            practicekey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practicekey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practicekey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practicekey.status == STARTED and not waitOnFlip:
            theseKeys = practicekey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _practicekey_allKeys.extend(theseKeys)
            if len(_practicekey_allKeys):
                practicekey.keys = _practicekey_allKeys[-1].name  # just the last key pressed
                practicekey.rt = _practicekey_allKeys[-1].rt
                practicekey.duration = _practicekey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime(format='float'))
    # check responses
    if practicekey.keys in ['', [], None]:  # No response was made
        practicekey.keys = None
    thisExp.addData('practicekey.keys',practicekey.keys)
    if practicekey.keys != None:  # we had a response
        thisExp.addData('practicekey.rt', practicekey.rt)
        thisExp.addData('practicekey.duration', practicekey.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('practice conditions.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
        fixation.setText('+')
        cue.setOri(cueOri)
        # Run 'Begin Routine' code from cuedrawcode
        import random
        cues = ['cueimages/cue1.png', 'cueimages/cue2.png'] 
        random_cue = random.choice(cues)
        cue.setImage(random_cue)  
        
        cue.flipHoriz = (cueOri == 180)
        cue.ori = 0
        
        
        #if cueOri is not None and cueOri != '':
            #cue.ori = float(cueOri)
            #cue.flipHoriz = (float(cueOri) == 180)
        #else:
            #cue.ori = 0
            #cue.flipHoriz = False  
        
            
        
        
        
        target.setPos([targetX, 0])
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        trial_counter.reset()
        trial_counter.setText(str(trials.thisN+1) + '/' + str(trials.nTotal))
        # Run 'Begin Routine' code from trialtiming
        rtClock = core.Clock()  
        
        # keep track of which components have finished
        trialComponents = [fixation, cue, target, key_resp, reminder, trial_counter]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *cue* updates
            
            # if cue is starting this frame...
            if cue.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                cue.frameNStart = frameN  # exact frame index
                cue.tStart = t  # local t and not account for scr refresh
                cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue.started')
                # update status
                cue.status = STARTED
                cue.setAutoDraw(True)
            
            # if cue is active this frame...
            if cue.status == STARTED:
                # update params
                pass
            
            # if cue is stopping this frame...
            if cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue.tStartRefresh + 0.4-frameTolerance:
                    # keep track of stop time/frame for later
                    cue.tStop = t  # not accounting for scr refresh
                    cue.tStopRefresh = tThisFlipGlobal  # on global time
                    cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.stopped')
                    # update status
                    cue.status = FINISHED
                    cue.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 2.8-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['a','l'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *reminder* updates
            
            # if reminder is starting this frame...
            if reminder.status == NOT_STARTED and tThisFlip >= 2.8-frameTolerance:
                # keep track of start time/frame for later
                reminder.frameNStart = frameN  # exact frame index
                reminder.tStart = t  # local t and not account for scr refresh
                reminder.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reminder.started')
                # update status
                reminder.status = STARTED
                reminder.setAutoDraw(True)
            
            # if reminder is active this frame...
            if reminder.status == STARTED:
                # update params
                pass
            
            # if reminder is stopping this frame...
            if reminder.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reminder.tStartRefresh + 1.2-frameTolerance:
                    # keep track of stop time/frame for later
                    reminder.tStop = t  # not accounting for scr refresh
                    reminder.tStopRefresh = tThisFlipGlobal  # on global time
                    reminder.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reminder.stopped')
                    # update status
                    reminder.status = FINISHED
                    reminder.setAutoDraw(False)
            
            # *trial_counter* updates
            
            # if trial_counter is starting this frame...
            if trial_counter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_counter.frameNStart = frameN  # exact frame index
                trial_counter.tStart = t  # local t and not account for scr refresh
                trial_counter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_counter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_counter.started')
                # update status
                trial_counter.status = STARTED
                trial_counter.setAutoDraw(True)
            
            # if trial_counter is active this frame...
            if trial_counter.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from trialtiming
            if t >= 4.5:
                continueRoutine = False
            
            if key_resp.keys:  
                rt = rtClock.getTime()  
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from track_rt
        if key_resp.keys:
            rt = key_resp.rt
            thisExp.addData('reaction_time', rt)
        
        if congr == 1:
            valid_rts.append(key_resp.rt)   
        elif congr == 0:
            invalid_rts.append(key_resp.rt)  
        
        
        # Run 'End Routine' code from fbcode
        print("Key pressed:", key_resp.keys)
        if key_resp.keys:
            if targetX < 0:
                if 'a' in key_resp.keys:
                    feedbackTxt = 'Correct'
                else:
                    feedbackTxt = 'Incorrect'
        
            elif targetX > 0:
                if 'l' in key_resp.keys:
                    feedbackTxt = 'Correct'
                else:
                    feedbackTxt = 'Incorrect'
        else:
            feedbackTxt = 'No response'
        
        feedbackTxt1.setText(feedbackTxt)
        
        # Run 'End Routine' code from trialtiming
        feedbackTxt = "No response"
        
        if key_resp.keys:
            rt = rtClock.getTime()
        
            if rt > 3.7:
                feedbackTxt = "Too slow"
            elif (targetX < 0 and 'a' in key_resp.keys) or (targetX > 0 and 'l' in key_resp.keys):
                feedbackTxt = "Correct"
            else:
                feedbackTxt = "Incorrect"
        
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback1.started', globalClock.getTime(format='float'))
        feedbackTxt1.setText(feedbackTxt)
        trialClock.reset()
        trialClock.setText(str(trials.thisN+1) + '/' + str(trials.nTotal))
        # keep track of which components have finished
        feedback1Components = [feedbackTxt1, trialClock]
        for thisComponent in feedback1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedbackTxt1* updates
            
            # if feedbackTxt1 is starting this frame...
            if feedbackTxt1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedbackTxt1.frameNStart = frameN  # exact frame index
                feedbackTxt1.tStart = t  # local t and not account for scr refresh
                feedbackTxt1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedbackTxt1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedbackTxt1.started')
                # update status
                feedbackTxt1.status = STARTED
                feedbackTxt1.setAutoDraw(True)
            
            # if feedbackTxt1 is active this frame...
            if feedbackTxt1.status == STARTED:
                # update params
                pass
            
            # if feedbackTxt1 is stopping this frame...
            if feedbackTxt1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedbackTxt1.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    feedbackTxt1.tStop = t  # not accounting for scr refresh
                    feedbackTxt1.tStopRefresh = tThisFlipGlobal  # on global time
                    feedbackTxt1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedbackTxt1.stopped')
                    # update status
                    feedbackTxt1.status = FINISHED
                    feedbackTxt1.setAutoDraw(False)
            
            # *trialClock* updates
            
            # if trialClock is starting this frame...
            if trialClock.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trialClock.frameNStart = frameN  # exact frame index
                trialClock.tStart = t  # local t and not account for scr refresh
                trialClock.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trialClock, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trialClock.started')
                # update status
                trialClock.status = STARTED
                trialClock.setAutoDraw(True)
            
            # if trialClock is active this frame...
            if trialClock.status == STARTED:
                # update params
                pass
            
            # if trialClock is stopping this frame...
            if trialClock.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trialClock.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    trialClock.tStop = t  # not accounting for scr refresh
                    trialClock.tStopRefresh = tThisFlipGlobal  # on global time
                    trialClock.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trialClock.stopped')
                    # update status
                    trialClock.status = FINISHED
                    trialClock.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback1" ---
        for thisComponent in feedback1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "endofpractice" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('endofpractice.started', globalClock.getTime(format='float'))
    Startkey.keys = []
    Startkey.rt = []
    _Startkey_allKeys = []
    start1.reset()
    # keep track of which components have finished
    endofpracticeComponents = [instructText1, Startkey, start1]
    for thisComponent in endofpracticeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "endofpractice" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructText1* updates
        
        # if instructText1 is starting this frame...
        if instructText1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructText1.frameNStart = frameN  # exact frame index
            instructText1.tStart = t  # local t and not account for scr refresh
            instructText1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructText1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructText1.started')
            # update status
            instructText1.status = STARTED
            instructText1.setAutoDraw(True)
        
        # if instructText1 is active this frame...
        if instructText1.status == STARTED:
            # update params
            pass
        
        # *Startkey* updates
        waitOnFlip = False
        
        # if Startkey is starting this frame...
        if Startkey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Startkey.frameNStart = frameN  # exact frame index
            Startkey.tStart = t  # local t and not account for scr refresh
            Startkey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Startkey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Startkey.started')
            # update status
            Startkey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Startkey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Startkey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Startkey.status == STARTED and not waitOnFlip:
            theseKeys = Startkey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Startkey_allKeys.extend(theseKeys)
            if len(_Startkey_allKeys):
                Startkey.keys = _Startkey_allKeys[-1].name  # just the last key pressed
                Startkey.rt = _Startkey_allKeys[-1].rt
                Startkey.duration = _Startkey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *start1* updates
        
        # if start1 is starting this frame...
        if start1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start1.frameNStart = frameN  # exact frame index
            start1.tStart = t  # local t and not account for scr refresh
            start1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start1.started')
            # update status
            start1.status = STARTED
            start1.setAutoDraw(True)
        
        # if start1 is active this frame...
        if start1.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endofpracticeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "endofpractice" ---
    for thisComponent in endofpracticeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('endofpractice.stopped', globalClock.getTime(format='float'))
    # check responses
    if Startkey.keys in ['', [], None]:  # No response was made
        Startkey.keys = None
    thisExp.addData('Startkey.keys',Startkey.keys)
    if Startkey.keys != None:  # we had a response
        thisExp.addData('Startkey.rt', Startkey.rt)
        thisExp.addData('Startkey.duration', Startkey.duration)
    thisExp.nextEntry()
    # the Routine "endofpractice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials1 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('mainconditions.csv'),
        seed=None, name='trials1')
    thisExp.addLoop(trials1)  # add the loop to the experiment
    thisTrials1 = trials1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials1.rgb)
    if thisTrials1 != None:
        for paramName in thisTrials1:
            globals()[paramName] = thisTrials1[paramName]
    
    for thisTrials1 in trials1:
        currentLoop = trials1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials1.rgb)
        if thisTrials1 != None:
            for paramName in thisTrials1:
                globals()[paramName] = thisTrials1[paramName]
        
        # --- Prepare to start Routine "trial1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial1.started', globalClock.getTime(format='float'))
        cueimg.setOri(cueOri)
        cueimg.setImage(random_cue)
        # Run 'Begin Routine' code from cuedrawcode_1
        import random
        cues = ['cueimages/cue1.png', 'cueimages/cue2.png'] 
        random_cue = random.choice(cues)
        cueimg.setImage(random_cue)  
        
        cueimg.flipHoriz = (cueOri == 180)
        cueimg.ori = 0
        
        
        
        
        
        target1.setPos([targetX, 0])
        target1.setImage('targetimage/circle_grey.png')
        key_resp2.keys = []
        key_resp2.rt = []
        _key_resp2_allKeys = []
        # Run 'Begin Routine' code from remindercode
        reminder1_shown = False
        reminder_time = iti + 0.4 + soa + 1.0  
        
        
        
        # keep track of which components have finished
        trial1Components = [fixation1, cueimg, target1, key_resp2, reminder1]
        for thisComponent in trial1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation1* updates
            
            # if fixation1 is starting this frame...
            if fixation1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation1.frameNStart = frameN  # exact frame index
                fixation1.tStart = t  # local t and not account for scr refresh
                fixation1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation1.started')
                # update status
                fixation1.status = STARTED
                fixation1.setAutoDraw(True)
            
            # if fixation1 is active this frame...
            if fixation1.status == STARTED:
                # update params
                pass
            
            # if fixation1 is stopping this frame...
            if fixation1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation1.tStartRefresh + iti-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation1.tStop = t  # not accounting for scr refresh
                    fixation1.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation1.stopped')
                    # update status
                    fixation1.status = FINISHED
                    fixation1.setAutoDraw(False)
            # Run 'Each Frame' code from fixationcode
            try:
                fix_duration = iti
            except:
                fix_duration = 0.8  
            
            
            # *cueimg* updates
            
            # if cueimg is starting this frame...
            if cueimg.status == NOT_STARTED and tThisFlip >= iti-frameTolerance:
                # keep track of start time/frame for later
                cueimg.frameNStart = frameN  # exact frame index
                cueimg.tStart = t  # local t and not account for scr refresh
                cueimg.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cueimg, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cueimg.started')
                # update status
                cueimg.status = STARTED
                cueimg.setAutoDraw(True)
            
            # if cueimg is active this frame...
            if cueimg.status == STARTED:
                # update params
                pass
            
            # if cueimg is stopping this frame...
            if cueimg.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cueimg.tStartRefresh + 0.4-frameTolerance:
                    # keep track of stop time/frame for later
                    cueimg.tStop = t  # not accounting for scr refresh
                    cueimg.tStopRefresh = tThisFlipGlobal  # on global time
                    cueimg.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cueimg.stopped')
                    # update status
                    cueimg.status = FINISHED
                    cueimg.setAutoDraw(False)
            
            # *target1* updates
            
            # if target1 is starting this frame...
            if target1.status == NOT_STARTED and tThisFlip >= iti + 0.4 + soa-frameTolerance:
                # keep track of start time/frame for later
                target1.frameNStart = frameN  # exact frame index
                target1.tStart = t  # local t and not account for scr refresh
                target1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target1.started')
                # update status
                target1.status = STARTED
                target1.setAutoDraw(True)
            
            # if target1 is active this frame...
            if target1.status == STARTED:
                # update params
                pass
            
            # if target1 is stopping this frame...
            if target1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target1.tStartRefresh + 2.8-frameTolerance:
                    # keep track of stop time/frame for later
                    target1.tStop = t  # not accounting for scr refresh
                    target1.tStopRefresh = tThisFlipGlobal  # on global time
                    target1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target1.stopped')
                    # update status
                    target1.status = FINISHED
                    target1.setAutoDraw(False)
            
            # *key_resp2* updates
            waitOnFlip = False
            
            # if key_resp2 is starting this frame...
            if key_resp2.status == NOT_STARTED and tThisFlip >= iti + 0.4 + soa-frameTolerance:
                # keep track of start time/frame for later
                key_resp2.frameNStart = frameN  # exact frame index
                key_resp2.tStart = t  # local t and not account for scr refresh
                key_resp2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp2.started')
                # update status
                key_resp2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp2.getKeys(keyList=['a','l'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp2_allKeys.extend(theseKeys)
                if len(_key_resp2_allKeys):
                    key_resp2.keys = _key_resp2_allKeys[-1].name  # just the last key pressed
                    key_resp2.rt = _key_resp2_allKeys[-1].rt
                    key_resp2.duration = _key_resp2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *reminder1* updates
            
            # if reminder1 is active this frame...
            if reminder1.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from remindercode
            #if not reminder1_shown and t >= reminder_time:
               # if not key_resp.keys:  
                   # reminder1.setAutoDraw(True)
                   # reminder1_shown = True
            
            #if key_resp.keys and reminder1_shown:
                #reminder1.setAutoDraw(False)
            
            
            if not reminder1_shown and t >= reminder_time:
                if not key_resp2.keys:
                    print(">>> Reminder SHOWN!")
                    reminder1.setAutoDraw(True)
                    reminder1_shown = True
            
            if key_resp2.keys and reminder1_shown:
                print(">>> Reminder HIDDEN after response.")
                reminder1.setAutoDraw(False)
            
            
            
            if t > reminder_time + 1.5 and not reminder1_shown:
                print(">>> Reminder should have shown by now, but didn't!")
            
            # Run 'Each Frame' code from timingcode
            if t >= 4.5:
                continueRoutine = False
            
            if key_resp.keys:  
                rt = rtClock.getTime()  
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial1" ---
        for thisComponent in trial1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial1.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp2.keys in ['', [], None]:  # No response was made
            key_resp2.keys = None
        trials1.addData('key_resp2.keys',key_resp2.keys)
        if key_resp2.keys != None:  # we had a response
            trials1.addData('key_resp2.rt', key_resp2.rt)
            trials1.addData('key_resp2.duration', key_resp2.duration)
        # Run 'End Routine' code from track_rt1
        #if congr == 1:
            #valid_rts.append(key_resp.rt)   
        #elif congr == 0:
            #invalid_rts.append(key_resp.rt)  
        
        #if key_resp2.rt is not None:
            #try:
                #rt_value = float(key_resp2.rt) 
                #if congr == 1:
                    #valid_rts.append(rt_value)
                #elif congr == 0:
                    #invalid_rts.append(rt_value)
            #except:
                #print("RT could not be converted:", key_resp2.rt)
        
        
        if key_resp2.rt is not None:
            try:
                rt_value = float(key_resp2.rt)
                if congr == 1:
                    valid_rts.append(rt_value)
                    print("✅ Valid RT added:", rt_value)
                elif congr == 0:
                    invalid_rts.append(rt_value)
                    print("✅ Invalid RT added:", rt_value)
            except:
                print("❌ Bad RT, not added:", key_resp2.rt)
        else:
            print("⚠️ No RT recorded this trial.")
        
        # Run 'End Routine' code from remindercode
        reminder1.setAutoDraw(False)
        
        
        
        # the Routine "trial1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials1'
    
    
    # --- Prepare to start Routine "finish" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('finish.started', globalClock.getTime(format='float'))
    finishMsg_2.setText('')
    # Run 'Begin Routine' code from endingfb
    if valid_rts:
        valid_av = sum(valid_rts) / len(valid_rts)
    else:
        valid_av = 0.0
    
    if invalid_rts:
        invalid_av = sum(invalid_rts) / len(invalid_rts)
    else:
        invalid_av = 0.0
    
    thisExp.addData('valid_av', valid_av)
    thisExp.addData('invalid_av', invalid_av)
    
    endfb = (
        "End of experiment.\n\n"
        f"Your average reaction time when the cue matched the target location trials was {int(valid_av * 1000)} ms.\n"
        f"When the cue pointed away from the target, it was {int(invalid_av * 1000)} ms.\n\n"
        "Thank you for participating!"
    )
    
    finishMsg_2.setText(endfb)
    
    # keep track of which components have finished
    finishComponents = [finishMsg_2]
    for thisComponent in finishComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "finish" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *finishMsg_2* updates
        
        # if finishMsg_2 is starting this frame...
        if finishMsg_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            finishMsg_2.frameNStart = frameN  # exact frame index
            finishMsg_2.tStart = t  # local t and not account for scr refresh
            finishMsg_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(finishMsg_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'finishMsg_2.started')
            # update status
            finishMsg_2.status = STARTED
            finishMsg_2.setAutoDraw(True)
        
        # if finishMsg_2 is active this frame...
        if finishMsg_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in finishComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "finish" ---
    for thisComponent in finishComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('finish.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from endingfb
    if congr == 1:
        valid_rts.append(key_resp.rt)
    elif congr == 0:
        invalid_rts.append(key_resp.rt)
    
    thisExp.nextEntry()
    # the Routine "finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
