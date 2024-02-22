#pragma once


namespace thoracuda {
namespace rangequery {

  struct __align__(16) Row {
    float x;
    float y;
    float t;
    int id;
  };

}
}