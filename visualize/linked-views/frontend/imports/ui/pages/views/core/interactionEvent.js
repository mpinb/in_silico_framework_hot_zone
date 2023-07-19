import { deepCopy } from "./utilCore";

class InteractionEvent {
    constructor(source_view_name, source_view_table) {
        this.source_view_name = source_view_name;
        this.source_view_table = source_view_table;
        this.views_target = [];
        this.viewManager = undefined;
    }

    setViewManager(viewManager){
        this.viewManager = viewManager;
    }

    isTargetView(viewName){
        return this.views_target.indexOf(viewName) != -1;
    }

    satisfiesNumSelectedRange(numSelectedRange) {
        return false;
    }

    satisfiesInteractionType(typeName) {
        return false;
    }
}

export class SelectionEvent extends InteractionEvent {   
    constructor(source_view_name, source_view_table) {
        super(source_view_name, source_view_table);
        
        this.indices = new Set();
        this.mappedIndices = {} // target table -> indices

        this.isDeselect = false;
        this.isUpdate = false;
        this.isSelect = false;        

        this.views_assign = [];
        this.views_intersect = [];
        this.views_union = [];
        this.views_clear = [];
    }

    setDeselect() {
        this.isDeselect = true;
        return this;
    }

    setIndices(indices, isUpdate = false) {
        this.indices = new Set(indices);

        this.isUpdate = isUpdate;
        this.isSelect = true;
        return this;
    }

    setViewsForOperations(viewsAssign, viewsIntersect, viewsUnion, viewsClear) {        
        this.viewsAssign = viewsAssign;
        this.viewsIntersect = viewsIntersect;
        this.viewsUnion = viewsUnion;
        this.viewsClear = viewsClear;

        this.views_target = [...viewsAssign, ...viewsIntersect, ...viewsUnion, ...viewsClear];
    }

    /**
     * Check filter conditions
     * 
     * 
     * 
     */
    satisfiesNumSelectedRange(numSelectedRange) {
        if(Array.isArray(numSelectedRange)){
            return this.indices.size >= numSelectedRange[0] && this.indices.size <= numSelectedRange[1];
        } else if (Number.isInteger(numSelectedRange)) {
            return this.indices.size == numSelectedRange;
        } else {
            throw Error(numSelectedRange);
        }
    }

    satisfiesInteractionType(typeName) {
        if(typeName == "deselect"){
            return this.isDeselect;
        } else {
            throw Error(typeName);
        }
    }

    /**
     * Apply operations
     * 
     * 
     * 
     */
    applyOperations(viewName, viewTable, currentIndices){

        let indicesForOp = this.indices;

        if(viewTable != this.source_view_table){
            /*
            console.log(viewTable, this.source_view_table);
            if(this.source_view_table == "synapses.csv"){                
                const synapsesTable = this.viewManager.dataManager.cacheExpanded["synapses.csv"]
                const neuronsTable = this.viewManager.dataManager.cacheExpanded["neurons.csv"]
                indicesForOp = synapsesTable.reduce((accum, row) => {
                    console.log(row.rowIdx);
                    if(this.indices.has(row.rowIdx)){
                        let neuronsRow = neuronsTable.find(r => r.neuron_id == row.neuron_id)
                        console.log(neuronsRow);
                        if(neuronsRow){
                            accum.add(neuronsRow.rowIdx);
                        }
                    }
                    return accum;
                }, new Set())                
            } else {
                throw Error(this.source_view_table);
            }
            */
        }

        if(this.viewsAssign.indexOf(viewName) !== -1){            
            return Array.from(indicesForOp);
        }

        if(this.viewsIntersect.indexOf(viewName) !== -1){
            if(!currentIndices.length){
                return Array.from(indicesForOp);
            }
            return currentIndices.filter(idx => indicesForOp.has(idx));
        }

        if(this.viewsUnion.indexOf(viewName) !== -1){               
            const unionSet = new Set([...indicesForOp, ...currentIndices])
            const union = Array.from(unionSet);            
            return union;
        }

        if(this.viewsClear.indexOf(viewName) !== -1){            
            return [];
        }
    }    
}


