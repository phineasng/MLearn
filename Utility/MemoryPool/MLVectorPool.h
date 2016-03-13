#ifndef MLEARN_VECTOR_MEMORY_POOL_INCLUDE
#define MLEARN_VECTOR_MEMORY_POOL_INCLUDE

// MLearn includes
#include <MLearn/Core>

// STL includes
#include <unordered_map>
#include <utility>

namespace MLearn{

	namespace Utility{

		namespace MemoryPool{

			template < typename ScalarType, typename Key = int >
			class MLVectorPool{
				static_assert( std::is_integral<Key>::value, "(MLVector) Memory pool needs an integer key type!" );
			public:
				typedef ScalarType 										Scalar;
				typedef MLVector<ScalarType> 							AllocatedType;
				typedef std::pair< AllocatedType, bool >				MappedType;
				typedef Key 											KeyType;
				typedef std::unordered_multimap< KeyType, MappedType >	PoolType;
				// Interface class 
				class PreallocatedMLVector{
				public:
					inline AllocatedType& getReference() const { return pooled_element.first; }
					~PreallocatedMLVector(){
						pooled_element.second = true;
					}
				protected:
					explicit PreallocatedMLVector( MappedType& refPair ): pooled_element(refPair){}
				private:	
					MappedType& pooled_element;
					friend PreallocatedMLVector MLVectorPool::get( KeyType key );
				};
			public:	
				static PreallocatedMLVector get( KeyType key ){
					MLEARN_ASSERT( key >= 0, "Input value must be a positive number!" );
					auto range = pool.equal_range(key);
					for ( auto it = range.first; it != range.second; ++it ){
						if ( it->second.second && ( it->second.first.size() == key) ){
							it->second.second = false;
							return PreallocatedMLVector( (it->second) );
						}
					}
					auto it = pool.emplace( key, std::make_pair< AllocatedType,bool >(AllocatedType(key), false) );
					return PreallocatedMLVector( (it->second) );
				} 
				static inline void empty(){
					pool.empty();
				}
			protected:
				MLVectorPool() = default;
			private:
				static PoolType pool;
			};

			template < typename T1, typename T2 >
			typename MLVectorPool<T1,T2>::PoolType MLVectorPool<T1,T2>::pool = typename MLVectorPool<T1,T2>::PoolType();

		}

	}

}

#endif