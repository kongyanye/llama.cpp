include(FetchContent)

set(TARGET llama-shard1)

set(TARGET_SRCS
    main.cpp
)

set(TARGET_DEFS
    ${LLAMA_DEFINITIONS}
)

set(TARGET_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include_directories(${TARGET_INCLUDES})

add_executable(${TARGET} ${TARGET_SRCS})

target_link_libraries(${TARGET} PRIVATE common llama ${LLAMA_EXTRA_LIBS})
target_compile_definitions(${TARGET} PRIVATE ${TARGET_DEFS})
target_compile_options(${TARGET} PRIVATE ${LLAMA_COMPILE_OPTIONS})

install(TARGETS ${TARGET} RUNTIME)