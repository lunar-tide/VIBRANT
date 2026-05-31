#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import lldb
import os
import datetime
import time

ignore_function = ['__do_global_dtors_aux','deregister_tm_clones','_fini','_init','_start','__stat','__fstat',
                   '_dl_relocate_static_pie']

def getFunctionMemoryDump(debugger: lldb.SBDebugger, command, result, internal_dict):
    args = command.split()
    breakFunction = args[0]
    dumpPath = args[1]
    step = 1
    interpreter = lldb.debugger.GetCommandInterpreter()
    returnObject = lldb.SBCommandReturnObject()
    interpreter.HandleCommand(f'b {breakFunction}', returnObject)
    output = returnObject.GetOutput()
    if "address" not in output:
        print(f"ERROR: The binary file not have a {breakFunction} function", sys.stderr)
        return
    debugger.HandleCommand('run')
    target: lldb.SBTarget = debugger.GetSelectedTarget()

    process: lldb.SBProcess = target.GetProcess()
    thread: lldb.SBThread = process.GetSelectedThread()
    threadName = thread.GetName()
    while process.is_alive:
        frame = thread.GetFrameAtIndex(0)
        current_name = frame.GetFunctionName()
        if current_name == breakFunction:
            memRegionList: lldb.SBMemoryRegionInfoList = process.GetMemoryRegions()
            memRegionInfo = lldb.SBMemoryRegionInfo()
            for i in range(memRegionList.GetSize()):
                memRegionList.GetMemoryRegionAtIndex(i, memRegionInfo)
                if memRegionInfo.GetName() == "[stack]":
                    stack_base = memRegionInfo.GetRegionBase()
                    stack_end = memRegionInfo.GetRegionEnd()
                    err = lldb.SBError()
                    stack_memory = process.ReadMemory(stack_base, stack_end - stack_base, err)
                    file_path = f"{dumpPath}/{threadName}/{breakFunction}/stack/{step}/base_{stack_base}_end_{stack_end}"
                    directory = os.path.dirname(file_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    with open(file_path, 'wb') as file:
                        file.write(stack_memory)
                if memRegionInfo.GetName() == "[heap]":
                    heap_base = memRegionInfo.GetRegionBase()
                    heap_end = memRegionInfo.GetRegionEnd()
                    err = lldb.SBError()
                    heap_memory = process.ReadMemory(heap_base, heap_end - heap_base, err)
                    file_path = f"{dumpPath}/{threadName}/{breakFunction}/heap/{step}/base_{heap_base}_end_{heap_end}"
                    directory = os.path.dirname(file_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    with open(file_path, 'wb') as file:
                        file.write(heap_memory)
            thread.StepOver()
            step += 1
        else:
            thread.StepOut()

def getAllSetpMemoryDump(debugger: lldb.SBDebugger, command, result, internal_dict):
    # 判断是否
    cwd = os.getcwd()
    print("cwd: " + cwd)
    cwddirlist = os.listdir(cwd)

    if "dump" in cwddirlist:
        pass
    else:
        os.mkdir("dump")

    allstep = 1
    debugger.HandleCommand('b main')
    debugger.HandleCommand('run')
    # 获取target：与被调试的可执行文件有关
    target: lldb.SBTarget = debugger.GetSelectedTarget()

    process: lldb.SBProcess = target.GetProcess()
    # 目前只能处理单一线程
    thread: lldb.SBThread = process.GetSelectedThread()

    while process.is_alive:
        print("Step Over times:" + str(allstep))

        threadName = thread.GetName()
        memRegionList: lldb.SBMemoryRegionInfoList = process.GetMemoryRegions()

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        file_path = 'dump/{}-{}-{}'.format(threadName, allstep, now_time)  # 设置输出文件路径
        file = open(file_path, mode='wb+')

        memRegionInfo = lldb.SBMemoryRegionInfo()
        print("Memory Region " + str(memRegionList.GetSize()))
        for i in range(memRegionList.GetSize()):
            memRegionList.GetMemoryRegionAtIndex(i, memRegionInfo)
            regionName = memRegionInfo.GetName()
            rbase = memRegionInfo.GetRegionBase()
            rend = memRegionInfo.GetRegionEnd()
            err = lldb.SBError()
            content = process.ReadMemory(rbase, rend - rbase, err)
            if (content != None):
                if regionName == None:  # 如果region名是None的话直接dump
                    file.write(content)
                    print("dump region: " + str(regionName))
                elif ".so" not in regionName:  # 如果region名不保护.so说明不是共享库，直接dump
                    file.write(content)
                    print("dump region: " + str(regionName))
                else:  # 否则说明regionName中含有.so，是共享库，不dump
                    continue

        file.close()
        thread.StepOver()
        allstep = allstep + 1
    pass


def is_system_library_function(frame):
    symbol = frame.GetSymbol()
    if symbol is None:
        return False

    symbol_name = symbol.GetName()
    if symbol_name is None:
        return False

    module = frame.GetModule()
    if module is None:
        return False
    module_name = module.GetFileSpec().GetFilename()
    if module_name is None:
        return False
    if 'lib' in module_name or 'system' in symbol_name or '___lldb' in symbol_name:
        return True
    return False

def getExecutingFunctionName(debugger: lldb.SBDebugger, command, result, internal_dict):
    target: lldb.SBTarget = debugger.GetSelectedTarget()
    function_names = []
    for module in target.modules:
        for symbol in module.symbols:
            if symbol.IsValid() and symbol.GetType() == lldb.eSymbolTypeCode:
                function_names.append(symbol.GetName())
    debugger.HandleCommand(f'b {command}')
    debugger.HandleCommand('run')
    interpreter = lldb.debugger.GetCommandInterpreter()
    returnObject = lldb.SBCommandReturnObject()

    for func in function_names:
        if ('_lldb_' in func) or ('lib' in func) or (func in ignore_function):
            continue
        else:
            interpreter.HandleCommand(f'b {func}',returnObject)
    target: lldb.SBTarget = debugger.GetSelectedTarget()
    run_function_names = []
    process: lldb.SBProcess = target.GetProcess()
    while True:
        state = process.GetState()
        if state == lldb.eStateStopped:
            thread: lldb.SBThread = process.GetSelectedThread()
            frame = thread.GetFrameAtIndex(0)
            current_function_name = frame.GetFunctionName()
            if (current_function_name in function_names) and (current_function_name not in run_function_names)\
                    and (current_function_name not in ignore_function):
                run_function_names.append(current_function_name)
                print(f"FuncName:{current_function_name}")
                process.Continue()
            else:
                process.Continue()
        elif state == lldb.eStateRunning:
            continue
        elif state == lldb.eStateExited:
            break
        elif state == lldb.eStateCrashed:
            break



def countInstructionSteps(debugger: lldb.SBDebugger, command, result, internal_dict):
    instruction_count = 1
    interpreter = lldb.debugger.GetCommandInterpreter()
    returnObject = lldb.SBCommandReturnObject()
    interpreter .HandleCommand(f'b {command}',returnObject)
    output = returnObject.GetOutput()
    if "address" not in output:
        print(f"ERROR: The binary file not have a {command} function",sys.stderr)
        return
    debugger.HandleCommand('run')

    target: lldb.SBTarget = debugger.GetSelectedTarget()

    process: lldb.SBProcess = target.GetProcess()
    thread: lldb.SBThread = process.GetSelectedThread()
    while process.is_alive:
        thread.StepInto()
        frame = debugger.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
        libTag = is_system_library_function(frame)
        if not libTag:
            frame1 = thread.GetFrameAtIndex(0)
            current_function_name = frame1.GetFunctionName()
            if current_function_name is not None and "___lldb" not in current_function_name and current_function_name not in ignore_function:
                instruction_count += 1
    print(f"Instruction Step: {instruction_count}")

def countFuncInstructionSteps(debugger: lldb.SBDebugger, command, result, internal_dict):
    instruction_count = 1
    interpreter = lldb.debugger.GetCommandInterpreter()
    returnObject = lldb.SBCommandReturnObject()
    interpreter.HandleCommand(f'b {command}', returnObject)
    output = returnObject.GetOutput()
    if "address" not in output:
        print(f"ERROR: The binary file not have a {command} function", sys.stderr)
        return
    debugger.HandleCommand('run')
    target: lldb.SBTarget = debugger.GetSelectedTarget()

    process: lldb.SBProcess = target.GetProcess()
    thread: lldb.SBThread = process.GetSelectedThread()

    while process.is_alive:
        thread.StepOver()
        frame = thread.GetFrameAtIndex(0)
        current_name = frame.GetFunctionName()
        if current_name == command:
            instruction_count += 1
        else:
            thread.StepOver()

    print(f"Func Instruction Step: {instruction_count}")

def executingStep(debugger: lldb.SBDebugger, command, result, internal_dict):
    thread = lldb.debugger.GetSelectedTarget().GetProcess().GetSelectedThread()
    steps = 187
    for i in range(steps):
        thread.StepOver()

def getFunctionName(debugger: lldb.SBDebugger, command, result, internal_dict):
    target: lldb.SBTarget = debugger.GetSelectedTarget()
    function_names = []
    for module in target.modules:
        for symbol in module.symbols:
            if symbol.IsValid() and symbol.GetType() == lldb.eSymbolTypeCode:
                function_names.append(symbol.GetName())
    print(f"Function Name: {function_names}")





def __lldb_init_module(debugger: lldb.SBDebugger, internal_dict):
    debugger.HandleCommand('command script add -f lldb_scripy.getAllSetpMemoryDump gASMD')
    debugger.HandleCommand('command script add -f lldb_scripy.countInstructionSteps countInstSteps')
    debugger.HandleCommand('command script add -f lldb_scripy.getExecutingFunctionName getExeFuncName')
    debugger.HandleCommand('command script add -f lldb_scripy.countFuncInstructionSteps countFuncInstSteps')
    debugger.HandleCommand('command script add -f lldb_scripy.executingStep execStep')
    debugger.HandleCommand('command script add -f lldb_scripy.getFunctionMemoryDump getFuncMemDump')
    debugger.HandleCommand('command script add -f lldb_scripy.getFunctionName getFuncName')
