#include <blust/backend/operations.hpp>

START_BLUST_NAMESPACE

// Global operation backend, initialized in 'blust.hpp' by init function
std::unique_ptr<operations> ops = std::unique_ptr<operations>();

END_BLUST_NAMESPACE