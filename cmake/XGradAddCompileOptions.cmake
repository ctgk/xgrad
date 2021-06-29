function(xgrad_add_compile_options target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE
        -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy
        -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op
        -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wswitch-default
        -Wundef -Werror -Wno-unused -Wconversion)
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0 -g3>)
    target_compile_definitions(${target} PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
endfunction(xgrad_add_compile_options)
