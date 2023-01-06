#ifndef __REDIS_HELPER__
#define __REDIS_HELPER__

#include <string.h>
#include "hiredis/hiredis.h"

namespace primihub {
class RedisHelper {
public:
  RedisHelper() {};
  ~RedisHelper();
  int connect(std::string redis_addr, const std::string &redis_passwd);
  void disconnect(void);
  redisContext *getContext(void);
private:
  redisContext *context_;
};

class RedisStringKVHelper : public RedisHelper {
public:
  int setString(const std::string &key, const std::string &value);
  int getString(const std::string &key, std::string &value);
};
} // namespace primihub

#endif
