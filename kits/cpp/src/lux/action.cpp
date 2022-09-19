#include "lux/action.hpp"

#include <string>
#include <type_traits>

#include "lux/exception.hpp"

namespace lux {
    UnitAction::UnitAction(UnitAction::RawType raw_) : Action(raw_) { populateMember(); }

    UnitAction::UnitAction(Type type_, Direction direction_, int64_t distance_, int64_t amount_, bool repeat_)
        : type(type_),
          direction(direction_),
          distance(distance_),
          amount(amount_),
          repeat(repeat_) {}

    UnitAction::UnitAction(Type type_, Direction direction_, Resource resource_, int64_t amount_, bool repeat_)
        : type(type_),
          direction(direction_),
          resource(resource_),
          amount(amount_),
          repeat(repeat_) {}

    UnitAction UnitAction::Move(Direction direction, bool repeat) {
        return UnitAction(Type::MOVE, direction, 1, 0, repeat);
    }

    UnitAction UnitAction::Transfer(Direction direction, Resource resource, int64_t amount, bool repeat) {
        return UnitAction(Type::TRANSFER, direction, resource, amount, repeat);
    }

    UnitAction UnitAction::Pickup(Resource resource, int64_t amount, bool repeat) {
        return UnitAction(Type::PICKUP, Direction::CENTER, resource, amount, repeat);
    }

    UnitAction UnitAction::Dig(bool repeat) { return UnitAction(Type::DIG, Direction::CENTER, 0, 0, repeat); }

    UnitAction UnitAction::SelfDestruct(bool repeat) {
        return UnitAction(Type::SELF_DESTRUCT, Direction::CENTER, 0, 0, repeat);
    }

    UnitAction UnitAction::Recharge(int64_t amount, bool repeat) {
        return UnitAction(Type::RECHARGE, Direction::CENTER, 0, amount, repeat);
    }

    void UnitAction::populateRaw() {
        raw[0] = std::underlying_type_t<UnitAction::Type>(type);
        raw[1] = std::underlying_type_t<UnitAction::Direction>(direction);
        raw[2] = distance;
        if (isTransferAction() || isPickupAction()) {
            raw[2] = std::underlying_type_t<UnitAction::Resource>(resource);
        }
        raw[3] = amount;
        raw[4] = repeat ? 1 : 0;
    }

    void UnitAction::populateMember() {
        if (raw[0] < 0 || raw[0] > 5) {
            throw lux::Exception("got invalid UnitAction type " + std::to_string(raw[0]));
        }
        if (raw[1] < 0 || raw[1] > 4) {
            throw lux::Exception("got invalid UnitAction direction " + std::to_string(raw[1]));
        }
        type      = static_cast<UnitAction::Type>(raw[0]);
        direction = static_cast<UnitAction::Direction>(raw[1]);
        if (isTransferAction() || isPickupAction()) {
            if (raw[2] < 0 || raw[2] > 5) {
                throw lux::Exception("got invalid UnitAction resource type " + std::to_string(raw[2]));
            }
            resource = static_cast<UnitAction::Resource>(raw[2]);
        }
        distance = raw[2];
        amount   = raw[3];
        repeat   = raw[4] != 0;
    }

    void to_json(json &j, const UnitAction a) {
        UnitAction copy = a;
        copy.toJson(j);
    }

    void from_json(const json &j, UnitAction &a) { a.fromJson(j); }

    FactoryAction::FactoryAction(FactoryAction::RawType raw_) : Action(raw_) { populateMember(); }

    FactoryAction::FactoryAction(Type type_) : Action(), type(type_) { populateRaw(); }

    FactoryAction FactoryAction::BuildLight() { return FactoryAction(Type::BUILD_LIGHT); }

    FactoryAction FactoryAction::BuildHeavy() { return FactoryAction(Type::BUILD_HEAVY); }

    FactoryAction FactoryAction::Water() { return FactoryAction(Type::WATER); }

    void FactoryAction::populateRaw() { raw = std::underlying_type_t<FactoryAction::Type>(type); }

    void FactoryAction::populateMember() {
        type = static_cast<FactoryAction::Type>(raw);
        if (!isBuildAction() && !isWaterAction()) {
            throw lux::Exception("got invalid FactoryAction type " + std::to_string(raw));
        }
    }

    void to_json(json &j, const FactoryAction a) {
        FactoryAction copy = a;
        copy.toJson(j);
    }

    void from_json(const json &j, FactoryAction &a) { a.fromJson(j); }

    BidAction::BidAction(std::string faction_, int64_t bid_) : faction(faction_), bid(bid_) {}

    SpawnAction::SpawnAction(std::array<int64_t, 2> spawn_, int64_t metal_, int64_t water_)
        : spawn(spawn_),
          metal(metal_),
          water(water_) {}
}  // namespace lux
