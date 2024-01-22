
include(FetchContent)

# Make catch2 available
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2)
FetchContent_MakeAvailable(catch)

# Tests need to be added as an executable
add_executable(tests test_projections.cpp test_pointsources.cpp)
target_link_libraries(tests PRIVATE thoracuda)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)
