find_program(LTI_CONFIG_CMD lti-local-config DOC "Path to the lti-local-config script. Usually under LTI_DIR/linux")
if(NOT LTI_CONFIG_CMD)
  message(FATAL_ERROR "Could not find the lti-local-config command. Set the LTI_CONFIG_CMD manually. Usually under LTI_DIR/linux")
endif()

# Ask ltilib about the cxx flags it needs
execute_process(COMMAND ${LTI_CONFIG_CMD} --cxxflags OUTPUT_VARIABLE LTI_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "\n" " " LTI_FLAGS "${LTI_FLAGS}")
string(REPLACE " " ";" LTI_FLAGS ${LTI_FLAGS})
message(STATUS "ltilib flags: ${LTI_FLAGS}")

# Ask ltilib about the libs it needs
execute_process(COMMAND ${LTI_CONFIG_CMD} --libs OUTPUT_VARIABLE LTI_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "\n" " " LTI_LIBS "${LTI_LIBS}")
string(REPLACE " " ";" LTI_LIBS ${LTI_LIBS})
message(STATUS "ltilib libs: ${LTI_LIBS}")

# Create interface target for lti lib
add_library(lti INTERFACE)

# Parse the options into cmake commands
foreach(FLAG ${LTI_FLAGS} ${LTI_LIBS})
  string(SUBSTRING "${FLAG}" 0 1 char0)
  string(SUBSTRING "${FLAG}" 1 1 OPTION)
  string(SUBSTRING "${FLAG}" 2 -1 VALUE)
  if(NOT char0 STREQUAL "-")
    message(FATAL_ERROR "Unknown flag ${FLAG}")
  endif()

  if("${OPTION}" STREQUAL "D")
    target_compile_definitions(lti INTERFACE ${VALUE})
  elseif("${OPTION}" STREQUAL "I")
    target_include_directories(lti INTERFACE ${VALUE})
  elseif("${OPTION}" STREQUAL "L")
    link_directories(${VALUE}) # Note: this is global because there is no target_link_directories
  elseif("${OPTION}" STREQUAL "l")
    target_link_libraries(lti INTERFACE ${VALUE}) # Note: this is global because there is no target_link_directories
  else()
    message(STATUS "Ignoring flag ${FLAG}")
  endif()
endforeach()
