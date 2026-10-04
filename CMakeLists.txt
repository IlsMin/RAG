cmake_minimum_required(VERSION 3.21)

project(RAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets Sql Network)
find_package(OpenMP REQUIRED)

set(FAISS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss)
set(OPENBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)

add_library(rag_faiss STATIC
    ${FAISS_ROOT}/faiss/Index.cpp
    ${FAISS_ROOT}/faiss/IndexFlat.cpp
    ${FAISS_ROOT}/faiss/IndexFlatCodes.cpp
    ${FAISS_ROOT}/faiss/IndexIDMap.cpp
    ${FAISS_ROOT}/faiss/IndexBinary.cpp
    ${FAISS_ROOT}/faiss/impl/FaissException.cpp
    ${FAISS_ROOT}/faiss/impl/AuxIndexStructures.cpp
    ${FAISS_ROOT}/faiss/impl/IDSelector.cpp
    ${FAISS_ROOT}/faiss/impl/CodePacker.cpp
    ${FAISS_ROOT}/faiss/utils/partitioning.cpp
    ${FAISS_ROOT}/faiss/utils/distances.cpp
    ${FAISS_ROOT}/faiss/utils/distances_simd.cpp
    ${FAISS_ROOT}/faiss/utils/extra_distances.cpp
    ${FAISS_ROOT}/faiss/utils/sorting.cpp
    ${FAISS_ROOT}/faiss/utils/utils.cpp
    ${FAISS_ROOT}/faiss/utils/random.cpp
    src/lapack_stub.cpp
)

target_include_directories(rag_faiss PUBLIC ${FAISS_ROOT})
target_compile_definitions(rag_faiss PUBLIC FINTEGER=int FAISS_MAIN_LIB)
target_link_libraries(rag_faiss PUBLIC
    OpenMP::OpenMP_CXX
    $<$<CONFIG:Debug>:${OPENBLAS_ROOT}/debug/openblas.lib>
    $<$<NOT:$<CONFIG:Debug>>:${OPENBLAS_ROOT}/release/openblas.lib>
)
if(MSVC)
    target_compile_options(rag_faiss PRIVATE /bigobj /EHsc /utf-8)
endif()

qt_standard_project_setup()

qt_add_executable(RAG
    main.cpp
    mainwindow.cpp
    mainwindow.h
    src/RagTypes.h
    src/TextChunker.h
    src/TextChunker.cpp
    src/LlamaClient.h
    src/LlamaClient.cpp
    src/KnowledgeBase.h
    src/KnowledgeBase.cpp
    src/RagPipeline.h
    src/RagPipeline.cpp
)

set_target_properties(RAG PROPERTIES WIN32_EXECUTABLE FALSE)

target_include_directories(RAG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(RAG
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Sql
        Qt6::Network
        rag_faiss
)

add_custom_command(TARGET RAG POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:Debug>:${OPENBLAS_ROOT}/debug/openblas.dll>$<$<NOT:$<CONFIG:Debug>>:${OPENBLAS_ROOT}/release/openblas.dll>
        $<TARGET_FILE_DIR:RAG>
    VERBATIM
)

if(MSVC)
    target_compile_options(RAG PRIVATE /utf-8 /EHsc)
    target_compile_definitions(RAG PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

install(TARGETS RAG
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET RAG
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
