def _amira_time_series_control(name = 'TimeSeriesControl', iconx = 10, icony = 100):
    return '''set hideNewModules 0
create HxDynamicFileSeriesCtrl "__NAME__"
"__NAME__" setIconPosition {iconx} {icony}
"__NAME__" setVar "CustomHelp" {{HxDynamicFileSeriesCtrl}}
"__NAME__" setVar "CustomXML" {{HxDynamicFileSeriesCtrl_config.xml}}
"__NAME__" time setMinMax 0 499
"__NAME__" time setSubMinMax 0 499
"__NAME__" time setValue 499
"__NAME__" time setDiscrete 1
"__NAME__" time setIncrement 1
"__NAME__" time animationMode -once
"__NAME__" time setAnimationDelay 10
"__NAME__" fire
"__NAME__" cachedSteps setMinMax 0 -16777216 16777216
"__NAME__" cachedSteps setValue 0 0
"__NAME__" options setValue 0 0
"__NAME__" options setToggleVisible 0 1
"__NAME__" options setValue 1 0
"__NAME__" options setToggleVisible 1 1
"__NAME__" options setValue 2 0
"__NAME__" options setToggleVisible 2 1
"__NAME__" options setValue 3 0
"__NAME__" options setToggleVisible 3 1
"__NAME__" applyTransformToResult 1
"__NAME__" fire
"__NAME__" setViewerMask 8191
"__NAME__" select
"__NAME__" setPickable 1'''.replace('__NAME__', name).format(iconx = iconx, icony = icony)

def _amira_get_files_template(name, rel_path_filelist, iconx = 10, icony = 100, time_series_control_name = 'TimeSeriesControl'):
    out = '''set hideNewModules 0
create HxDynamicFileSeriesCtrl "__NAME__"
"__NAME__" setIconPosition {iconx} {icony}
"__NAME__" setVar "CustomHelp" {{HxDynamicFileSeriesCtrl}}
"__NAME__" init -loadCmd {{load "$FILENAME"}} \\'''.format(iconx = iconx, icony = icony)

    for rel_path in rel_path_filelist[:-1]:
        out+= "${{SCRIPTDIR}}/{} \\\n".format(rel_path)
    out+= "${{SCRIPTDIR}}/{}".format(rel_path_filelist[-1])
    
    out += '''"__NAME__" cachedSteps setMinMax 0 0 500
"__NAME__" cachedSteps setValue 0 500
"__NAME__" options setValue 0 0
"__NAME__" options setToggleVisible 0 1
"__NAME__" options setValue 1 0
"__NAME__" options setToggleVisible 1 1
"__NAME__" options setValue 2 0
"__NAME__" options setToggleVisible 2 1
"__NAME__" options setValue 3 0
"__NAME__" options setToggleVisible 3 1
"__NAME__" fire
"__NAME__" time connect {time_series_control_name}
"__NAME__" time setMinMax 0 499
"__NAME__" time setSubMinMax 0 499
"__NAME__" time setValue 499
"__NAME__" time setDiscrete 1
"__NAME__" time setIncrement 1
"__NAME__" time animationMode -once
"__NAME__" time setAnimationDelay 0
"__NAME__" applyTransformToResult 1
"__NAME__" fire
"__NAME__" setViewerMask 16383
"__NAME__" setPickable 1'''.format(time_series_control_name = time_series_control_name)
    return out.replace('__NAME__',name) 

def _amira_landmark_view(input_name, color, size, name = 'LandmarkView', iconx = 10, icony = 30):
    return '''set hideNewModules 0
create HxDisplayLandmarks "{name}"
"__NAME__" setIconPosition {iconx} {icony}
"__NAME__" setVar "CustomHelp" {{HxDisplayLandmarks}}
"__NAME__" setColor {color_R} {color_G} {color_B}
"__NAME__" data connect "{input_name}"
"__NAME__" fire
"__NAME__" chooseSet setIndex 0 0
"__NAME__" display setValue 0 0
"__NAME__" display setToggleVisible 0 1
"__NAME__" drawStyle setValue 0 0
"__NAME__" drawStyle setToggleVisible 0 1
"__NAME__" drawStyle setValue 1 0
"__NAME__" drawStyle setToggleVisible 1 1
"__NAME__" size setMinMax 0 5
"__NAME__" size setButtons 0
"__NAME__" size setEditButton 1
"__NAME__" size setIncrement 0.333333
"__NAME__" size setValue 0.5
"__NAME__" size setSubMinMax 0 5
"__NAME__" complexity setMinMax 0 1
"__NAME__" complexity setButtons 0
"__NAME__" complexity setEditButton 1
"__NAME__" complexity setIncrement 0.1
"__NAME__" complexity setValue 0.25
"__NAME__" complexity setSubMinMax 0 1
"__NAME__" setBaseSize 321.51
"__NAME__" fire
"__NAME__" setViewerMask 16383
"__NAME__" setPickable 1'''.replace('__NAME__', name).format(name = name, iconx = iconx, icony = icony,
                                                            color_R = color[0],
                                                            color_G = color[1],
                                                            color_B = color[2])