/*
 Copyright 2022 Primihub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef SRC_PRIMIHUB_DATA_STORE_FACTORY_H_
#define SRC_PRIMIHUB_DATA_STORE_FACTORY_H_

#include <memory>
#include <boost/algorithm/string.hpp>
#include "src/primihub/data_store/driver.h"
#include "src/primihub/data_store/csv/csv_driver.h"
#include "src/primihub/data_store/sqlite/sqlite_driver.h"
#ifdef ENABLE_MYSQL_DRIVER
#include "src/primihub/data_store/mysql/mysql_driver.h"
#endif
#define CSV_DRIVER_NAME "CSV"
#define SQLITE_DRIVER_NAME "SQLITE"
#define HDFS_DRIVER_NAME "HDFS"
#define MYSQL_DRIVER_NAME "MYSQL"
namespace primihub {
class DataDirverFactory {
 public:
    using DataSetAccessInfoPtr = std::unique_ptr<DataSetAccessInfo>;
    static std::shared_ptr<DataDriver>
    getDriver(const std::string &dirverName, const std::string& nodeletAddr, DataSetAccessInfoPtr access_info = nullptr) {
        if (boost::to_upper_copy(dirverName) == CSV_DRIVER_NAME) {
            if (access_info == nullptr) {
                return std::make_shared<CSVDriver>(nodeletAddr);
            } else {
                return std::make_shared<CSVDriver>(nodeletAddr, std::move(access_info));
            }
        } else if (dirverName == HDFS_DRIVER_NAME) {
            // return new HDFSDriver(dirverName);
            // TODO not implemented yet
        } else if (boost::to_upper_copy(dirverName) == SQLITE_DRIVER_NAME) {
            if (access_info == nullptr) {
                return std::make_shared<SQLiteDriver>(nodeletAddr);
            } else {
                return std::make_shared<SQLiteDriver>(nodeletAddr, std::move(access_info));
            }
        }
#ifdef ENABLE_MYSQL_DRIVER
        else if (boost::to_upper_copy(dirverName) == MYSQL_DRIVER_NAME) {
            if (access_info == nullptr) {
                return std::make_shared<MySQLDriver>(nodeletAddr);
            } else {
                return std::make_shared<MySQLDriver>(nodeletAddr, std::move(access_info));
            }
        }
#endif
        else {
            std::string err_msg = "[DataDirverFactory]Invalid dirver name [" + dirverName + "]";
            throw std::invalid_argument(err_msg);
        }
    }

    static DataSetAccessInfoPtr createCSVAccessInfo(const std::string& file_path) {
        return std::make_unique<CSVAccessInfo>(file_path);
    }

    static DataSetAccessInfoPtr createSQLiteAccessInfo(const std::string& db_path,
            const std::string& tab_name, const std::vector<std::string>& query_index) {
        return std::make_unique<SQLiteAccessInfo>(db_path, tab_name, query_index);
    }

    static DataSetAccessInfoPtr createMySQLAccessInfo(const std::string& ip,
            uint32_t port, const std::string& user_name,
            const std::string& password, const std::string& database,
            const std::string& db_name, const std::string& table_name,
            const std::vector<std::string>& query_colums) {
        VLOG(5) << "ip: " << ip << " "
                <<  " port: " << port << " "
                << " user_name: " << user_name << " "
                << "password: " << password << " "
                << "database: " << database << " "
                << "db_name: " << db_name << " "
                << "table_name: " << table_name << " "
                << "query_colums size: " << query_colums.size();
#ifdef ENABLE_MYSQL_DRIVER
        return std::make_unique<MySQLAccessInfo>(ip, port, user_name,
                password, database, db_name, table_name, query_colums);
#else
        LOG(ERROR) << "MySQL is not enabled";
        return nullptr;
#endif
    }


};

} // namespace primihub

#endif // SRC_PRIMIHUB_DATA_STORE_FACTORY_H_
